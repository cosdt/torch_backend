#include <deque>
#include <vector>

#include <c10/util/CallOnce.h>
#include "csrc/backend/Context.h"

namespace c10::backend {
namespace {

/*
 * Currently, there is one device properties pool containing the information and
 * capability about each compute-device.
 *
 * Device properties are lazily initialized when the first time properties are
 * requested for a device.
 */
c10::DeviceIndex num_devices = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_prop_flags;
std::vector<DeviceProp> device_properties;

void initContextVectors() {
  num_devices = c10::backend::device_count();
  device_prop_flags.resize(num_devices);
  device_properties.resize(num_devices);
}

void initDeviceProperty(c10::DeviceIndex device) {
  c10::backend::get_device_properties(&device_properties[device], device);
}

inline void check_device(c10::DeviceIndex device) {
  TORCH_CHECK(
      device >= 0 && device < num_devices,
      "device is out of range, device is ",
      static_cast<int>(device),
      ", total number of device is ",
      static_cast<int>(num_devices),
      ".");
}
} // anonymous namespace

DeviceProp* getDeviceProperties(c10::DeviceIndex device) {
  c10::call_once(init_flag, initContextVectors);
  if (device == -1)
    device = c10::backend::current_device();
  check_device(device);
  c10::call_once(device_prop_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

} // namespace c10::backend
