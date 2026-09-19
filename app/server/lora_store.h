#pragma once

// Persistent LoRA adapter store for the WebUI.
//
// Adapters live in <models_root>/<model_directory>/loras/ so that a relative
// "loras/<file>" session option resolves against the model root the way every
// family already resolves relative adapter paths. Unlike /v1/ui/upload, which
// writes to a per-process temporary directory that is removed when the server
// exits, these survive restarts until deleted.
//
// Adapters are identified by reading the safetensors header rather than by
// filename. The header is a little-endian u64 length followed by that many
// bytes of JSON, so it can be read from the first few tens of KB of a file --
// locally, or over HTTP with a Range request against a remote that has not
// been downloaded. Tensor names are the authority: the `_comfyui` layouts in
// the public adapter repos are real LoRAs that this engine cannot consume, and
// only the tensor names distinguish them.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace minitts::server {

struct AdapterInfo {
    std::string file;          // basename
    std::string relative_path; // "loras/<file>", for the session option
    uint64_t bytes = 0;
    std::string branch;        // "ar", "nar", "both" or "unknown"
    bool loadable = false;     // unfused lora_A/lora_B pairs this engine accepts
    std::string layout;        // "unfused", "comfyui", "not-an-adapter"
    std::string rank;          // from __metadata__ when present
    std::string base_model;    // from __metadata__ when present
    std::string note;          // why it is not loadable, when it is not
};

// Parses a safetensors header (the JSON object) and classifies it.
AdapterInfo classify_safetensors_header(const std::string & header_json);

// Reads a local safetensors header. Returns nullopt when the file is not a
// readable safetensors container.
std::optional<std::string> read_local_safetensors_header(const std::filesystem::path & path);

// Reads a remote safetensors header with two ranged GETs. Returns nullopt when
// the remote refuses ranges or the response is not a safetensors container.
// Roughly 40 KB of transfer identifies a multi-hundred-megabyte adapter.
std::optional<std::string> read_remote_safetensors_header(const std::string & url);

// <model_directory>/loras, created on demand.
std::filesystem::path adapter_directory(const std::filesystem::path & model_directory);

// Every adapter already stored for a model.
std::vector<AdapterInfo> list_stored_adapters(const std::filesystem::path & model_directory);

struct RemoteAdapter {
    std::string file;      // path within the repo
    uint64_t bytes = 0;
    AdapterInfo info;      // from the ranged header probe
};

// Lists the safetensors in a Hugging Face repo and classifies each one from its
// header. `repo` is the short "<user>/<model>" form.
std::vector<RemoteAdapter> browse_remote_adapters(const std::string & repo, const std::string & revision);

// Downloads one file from a repo into <model_directory>/loras/, returning the
// stored path. Refuses to overwrite an existing file.
std::filesystem::path download_remote_adapter(
    const std::string & repo, const std::string & revision, const std::string & file,
    const std::filesystem::path & model_directory);

}  // namespace minitts::server
