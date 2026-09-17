#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

enum class TensorPlacement : std::uint8_t {
    Device,
    // Page-locked host memory mapped into the device address space. Kernels read it through the
    // same device pointer as a resident tensor, but the bytes never occupy VRAM. For parameters
    // that are not touched on every step.
    HostMapped,
    ValidateOnly,
};

struct ObjectHandle {
    std::size_t index = 0;
};

struct DeviceMaterialization {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
};

struct HostMaterialization {
    ObjectHandle object;
};

// A tensor held in mapped page-locked host memory rather than VRAM. `bytes` is the payload size
// recorded at bind time so the materializer can validate it against the artifact before copying.
struct HostMappedMaterialization {
    ObjectHandle object;
    std::uint64_t bytes = 0;
};

struct MaterializationPlan {
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::vector<DeviceMaterialization> device_objects;
    std::vector<HostMaterialization> host_objects;
    std::vector<HostMappedMaterialization> host_mapped_objects;
};

class Binder {
public:
    explicit Binder(const Reader& reader);

    ObjectHandle require_tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    [[nodiscard]] bool contains(std::string_view name) const noexcept;
    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    void materialize_on_device(ObjectHandle handle);
    void materialize_mapped_host(ObjectHandle handle);
    void retain_on_host(ObjectHandle handle);
    void validate_only(ObjectHandle handle);
    MaterializationPlan finish();

private:
    ObjectHandle find_unconsumed(std::string_view name);

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    MaterializationPlan materialization_;
};

} // namespace ninfer::artifact
