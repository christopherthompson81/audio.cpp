#include "lora_store.h"

#include "engine/framework/io/json.h"

#include "httplib.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <atomic>
#include <regex>
#include <thread>
#include <stdexcept>

namespace minitts::server {
namespace {

constexpr uint64_t kMaxHeaderBytes = 8u * 1024 * 1024;

std::string host_for(const std::string & url) {
    const auto scheme = url.find("://");
    if (scheme == std::string::npos) throw std::runtime_error("bad url: " + url);
    const auto rest = url.find('/', scheme + 3);
    return url.substr(0, rest == std::string::npos ? url.size() : rest);
}

std::string path_for(const std::string & url) {
    const auto scheme = url.find("://");
    const auto rest = url.find('/', scheme + 3);
    return rest == std::string::npos ? "/" : url.substr(rest);
}

httplib::Client make_client(const std::string & url) {
    httplib::Client client(host_for(url));
    client.set_follow_location(true);
    client.set_connection_timeout(30, 0);
    client.set_read_timeout(120, 0);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    client.enable_server_certificate_verification(true);
#endif
    return client;
}

httplib::Headers auth_headers() {
    httplib::Headers headers{{"User-Agent", "audio.cpp webui adapter store/1.0"}};
    if (const char * token = std::getenv("HF_TOKEN"); token != nullptr && *token != 0) {
        headers.emplace("Authorization", std::string("Bearer ") + token);
    }
    return headers;
}

// A basename that cannot escape the adapter directory.
std::string safe_basename(const std::string & file) {
    auto name = std::filesystem::path(file).filename().string();
    std::string cleaned;
    for (const char c : name) {
        cleaned += (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '_') ? c : '_';
    }
    if (cleaned.empty() || cleaned.front() == '.') cleaned = "adapter_" + cleaned;
    return cleaned;
}

}  // namespace

AdapterInfo classify_safetensors_header(const std::string & header_json) {
    AdapterInfo info;
    const auto root = engine::io::json::parse(header_json);
    bool has_lora_ab = false;    // lora_A / lora_B -- what this engine merges
    bool has_comfy = false;      // diffusion_model.* or lora_up / lora_down
    size_t tensors = 0;

    if (!root.is_object()) throw std::runtime_error("safetensors header is not a JSON object");
    for (const auto & [name, value] : root.as_object()) {
        if (name == "__metadata__") {
            // Both keys are PEFT conventions rather than any one family's.
            info.rank = engine::io::json::optional_string(value, "rank", "");
            info.base_model = engine::io::json::optional_string(value, "base_model", "");
            continue;
        }
        ++tensors;
        if (name.rfind("diffusion_model.", 0) == 0) has_comfy = true;
        if (name.find(".lora_up") != std::string::npos ||
            name.find(".lora_down") != std::string::npos) has_comfy = true;
        if (name.find(".lora_A") != std::string::npos ||
            name.find(".lora_B") != std::string::npos) has_lora_ab = true;
    }

    // Note what the file *is*, not what it suits. lora_A/lora_B is the shape
    // every LoRA-consuming family here reads, and a ComfyUI fused layout is one
    // none of them can read; which model an unfused adapter was trained against
    // is a question for the loader, which has the tensor layout to answer it.
    if (has_comfy) {
        info.layout = "comfyui";
        info.unfused_lora = false;
        info.note = "ComfyUI fused layout; this engine needs the unfused file";
    } else if (has_lora_ab) {
        info.layout = "unfused";
        info.unfused_lora = true;
    } else {
        info.layout = "not-an-adapter";
        info.unfused_lora = false;
        info.note = tensors == 0 ? "no tensors" : "no LoRA A/B pairs; a different component";
    }
    return info;
}

std::optional<std::string> read_local_safetensors_header(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    uint64_t length = 0;
    in.read(reinterpret_cast<char *>(&length), sizeof(length));
    if (!in || length == 0 || length > kMaxHeaderBytes) return std::nullopt;
    std::string header(static_cast<size_t>(length), '\0');
    in.read(header.data(), static_cast<std::streamsize>(length));
    if (!in) return std::nullopt;
    return header;
}

namespace {
// Shares one connection across the many probes a repo listing performs.
std::optional<std::string> read_remote_safetensors_header(httplib::Client & client, const std::string & target);
}  // namespace

std::optional<std::string> read_remote_safetensors_header(const std::string & url) {
    auto client = make_client(url);
    return read_remote_safetensors_header(client, path_for(url));
}

namespace {
std::optional<std::string> read_remote_safetensors_header(httplib::Client & client, const std::string & target) {
    // Nearly every adapter header fits well inside this, so one request is
    // usually enough; a larger one falls back to an exact second request.
    constexpr uint64_t kProbeBytes = 96u * 1024;
    auto headers = auth_headers();
    headers.emplace("Range", "bytes=0-" + std::to_string(kProbeBytes - 1));
    auto probe = client.Get(target, headers);
    if (!probe || (probe->status != 206 && probe->status != 200) || probe->body.size() < 8) {
        return std::nullopt;
    }
    uint64_t length = 0;
    std::memcpy(&length, probe->body.data(), sizeof(length));
    if (length == 0 || length > kMaxHeaderBytes) return std::nullopt;
    if (probe->body.size() >= 8 + length) {
        return probe->body.substr(8, static_cast<size_t>(length));
    }
    auto exact = auth_headers();
    exact.emplace("Range", "bytes=8-" + std::to_string(8 + length - 1));
    auto rest = client.Get(target, exact);
    if (!rest || (rest->status != 206 && rest->status != 200) || rest->body.size() < length) {
        return std::nullopt;
    }
    return rest->body.substr(0, static_cast<size_t>(length));
}

httplib::Client probe_client() {
    httplib::Client client("https://huggingface.co");
    client.set_follow_location(true);
    client.set_connection_timeout(30, 0);
    client.set_read_timeout(120, 0);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    client.enable_server_certificate_verification(true);
#endif
    return client;
}
}  // namespace

std::string sanitize_group(const std::string & group) {
    // One component, and only characters that cannot mean anything else to a
    // path: no separators, no "..", nothing hidden.
    auto name = std::filesystem::path(group).filename().string();
    std::string cleaned;
    for (const char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') cleaned += c;
    }
    return cleaned;
}

std::filesystem::path adapter_directory(const std::filesystem::path & model_directory,
                                        const std::string & group) {
    auto directory = model_directory;
    std::error_code ec;
    if (std::filesystem::is_regular_file(directory, ec)) directory = directory.parent_path();
    directory /= "loras";
    const auto cleaned = sanitize_group(group);
    if (!cleaned.empty()) directory /= cleaned;
    return directory;
}

namespace {

// One directory's worth of adapters, filed under `group` ("" for the flat root).
void collect_adapters(const std::filesystem::path & directory, const std::string & group,
                      std::vector<AdapterInfo> & found) {
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec)) return;
    for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        const auto path = it->path();
        if (path.extension() != ".safetensors") continue;
        auto header = read_local_safetensors_header(path);
        AdapterInfo info = header.has_value() ? classify_safetensors_header(*header) : AdapterInfo{};
        if (!header.has_value()) {
            info.layout = "unreadable";
            info.note = "could not read a safetensors header";
        }
        info.file = path.filename().string();
        info.group = group;
        info.relative_path = group.empty() ? "loras/" + info.file
                                           : "loras/" + group + "/" + info.file;
        info.bytes = static_cast<uint64_t>(std::filesystem::file_size(path, ec));
        found.push_back(std::move(info));
    }
}

}  // namespace

std::vector<AdapterInfo> list_stored_adapters(const std::filesystem::path & model_directory) {
    std::vector<AdapterInfo> found;
    const auto root = adapter_directory(model_directory);
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec)) return found;

    // The flat root stays readable so a file dropped in by hand still appears;
    // it simply has no group and the UI offers it everywhere.
    collect_adapters(root, "", found);

    // One level down, and no deeper: the folder name is the group. Symlinked
    // subdirectories are skipped -- the store is for files it was given.
    for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_symlink(ec) || !it->is_directory(ec)) continue;
        const auto name = it->path().filename().string();
        if (sanitize_group(name) != name) continue;
        collect_adapters(it->path(), name, found);
    }

    std::sort(found.begin(), found.end(), [](const AdapterInfo & a, const AdapterInfo & b) {
        return a.group == b.group ? a.file < b.file : a.group < b.group;
    });
    return found;
}

std::vector<RemoteAdapter> browse_remote_adapters(const std::string & repo, const std::string & revision) {
    static const std::regex kRepo(R"(^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$)");
    if (!std::regex_match(repo, kRepo)) {
        throw std::runtime_error("expected a short \"owner/model\" repo id, got: " + repo);
    }
    const std::string api = "https://huggingface.co/api/models/" + repo +
                            (revision.empty() ? "" : "/revision/" + revision) + "?blobs=true";
    auto client = make_client(api);
    auto listing = client.Get(path_for(api), auth_headers());
    if (!listing || listing->status != 200) {
        throw std::runtime_error("could not list " + repo +
                                 (listing ? " (HTTP " + std::to_string(listing->status) + ")" : ""));
    }
    const auto root = engine::io::json::parse(listing->body);

    // A repo with hundreds of safetensors would otherwise mean hundreds of round
    // trips inside one request; stop well before that.
    constexpr size_t kMaxProbes = 64;

    const auto * siblings = root.find("siblings");
    if (siblings == nullptr || !siblings->is_array()) {
        throw std::runtime_error("unexpected listing for " + repo + ": no file list");
    }

    std::vector<RemoteAdapter> adapters;
    for (const auto & sibling : siblings->as_array()) {
        if (adapters.size() >= kMaxProbes) break;
        const auto name = engine::io::json::optional_string(sibling, "rfilename", "");
        if (name.size() < 12 || name.substr(name.size() - 12) != ".safetensors") continue;
        RemoteAdapter adapter;
        adapter.file = name;
        if (const auto * size = sibling.find("size"); size != nullptr && size->is_number()) {
            adapter.bytes = static_cast<uint64_t>(size->as_number());
        }
        adapters.push_back(std::move(adapter));
    }

    // These probes are pure latency, so they run together. Each worker keeps its
    // own client because httplib clients are not shared across threads.
    const std::string prefix = "/" + repo + "/resolve/" + (revision.empty() ? "main" : revision) + "/";
    const size_t workers = std::min<size_t>(adapters.size(), 8);
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (size_t worker = 0; worker < workers; ++worker) {
        pool.emplace_back([&] {
            auto client = probe_client();
            for (size_t i = next.fetch_add(1); i < adapters.size(); i = next.fetch_add(1)) {
                auto & adapter = adapters[i];
                if (auto header = read_remote_safetensors_header(client, prefix + adapter.file);
                    header.has_value()) {
                    try {
                        adapter.info = classify_safetensors_header(*header);
                    } catch (const std::exception & error) {
                        adapter.info.layout = "unreadable";
                        adapter.info.note = error.what();
                    }
                } else {
                    adapter.info.layout = "unreadable";
                    adapter.info.note = "could not read the header over HTTP";
                }
                adapter.info.file = std::filesystem::path(adapter.file).filename().string();
                adapter.info.bytes = adapter.bytes;
            }
        });
    }
    for (auto & thread : pool) thread.join();

    return adapters;
}

std::filesystem::path download_remote_adapter(
    const std::string & repo, const std::string & revision, const std::string & file,
    const std::filesystem::path & model_directory, const std::string & group) {
    static const std::regex kRepo(R"(^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$)");
    if (!std::regex_match(repo, kRepo)) {
        throw std::runtime_error("expected a short \"owner/model\" repo id, got: " + repo);
    }
    const auto directory = adapter_directory(model_directory, group);
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    const auto destination = directory / safe_basename(file);
    if (std::filesystem::exists(destination, ec)) {
        throw std::runtime_error("already stored: " + destination.filename().string());
    }

    const std::string url = "https://huggingface.co/" + repo + "/resolve/" +
                            (revision.empty() ? "main" : revision) + "/" + file;
    // Staged under a partial name so an interrupted download is never mistaken
    // for a stored adapter by the listing.
    const auto staging = destination.string() + ".partial";
    std::ofstream out(staging, std::ios::binary);
    if (!out) throw std::runtime_error("could not write into " + directory.string());

    auto client = make_client(url);
    auto response = client.Get(path_for(url), auth_headers(),
        [&](const char * data, size_t size) {
            out.write(data, static_cast<std::streamsize>(size));
            return static_cast<bool>(out);
        });
    out.close();
    if (!response || response->status != 200) {
        std::filesystem::remove(staging, ec);
        throw std::runtime_error("download failed for " + file +
            (response ? " (HTTP " + std::to_string(response->status) + ")" : ""));
    }
    // Verify before publishing: a file no family here can read is worse than no
    // file. Whether it matches the model it was fetched for is the loader's call.
    if (auto header = read_local_safetensors_header(staging); header.has_value()) {
        const auto info = classify_safetensors_header(*header);
        if (!info.unfused_lora) {
            std::filesystem::remove(staging, ec);
            throw std::runtime_error("rejected " + file + ": " +
                (info.note.empty() ? "not an unfused adapter" : info.note));
        }
    } else {
        std::filesystem::remove(staging, ec);
        throw std::runtime_error("rejected " + file + ": not a safetensors file");
    }
    std::filesystem::rename(staging, destination, ec);
    if (ec) throw std::runtime_error("could not store " + destination.filename().string());
    return destination;
}

}  // namespace minitts::server
