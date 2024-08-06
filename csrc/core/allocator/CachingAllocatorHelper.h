#include <c10/core/Device.h>
#include <c10/util/irange.h>
namespace c10::backend::CachingAllocator {
static const int MEM_SUCCESS = 0;
static const int MEM_ALLOCATION_ERROR = 1;

/*
CachingAllocatorHelper contains functions that are used by
DefaultCachingAllocator, and each backend should have its own implementation.
*/
class CachingAllocatorHelper {
 public:
  // Wraps the insert event function
  virtual void insertEventWrapper(
      c10::DeviceIndex device,
      std::function<void()> insertEventFn) = 0;

  // Returns the current stream for the given device
  virtual void* getCurrentStream(c10::DeviceIndex) = 0;

  // Synchronizes the stream. e.g. cudaStreamSynchronize
  virtual int synchronizeStream(void* stream) = 0;

  // Wait for compute device to finish. e.g. cudaDeviceSynchronize.
  virtual void deviceSynchronize() = 0;

  /*
   memory management
   */

  // Frees memory on the device. e.g. cudaFree
  virtual int memFree(void* devPtr) = 0;

  // Allocate memory on the device. e.g. cudaMalloc
  virtual int memAlloc(void** devPtr, size_t size) = 0;

  // Gets free and total device memory. e.g. cudaMemGetInfo
  virtual int memGetInfo(size_t* free, size_t* total) = 0;

  // Free an address range reservation. e.g. cuMemAddressFree
  virtual int memAddressFree(void* ptr, size_t size) = 0;

  // Allocate an address range reservation. e.g. cuMemAddressReserve
  virtual int memAddressReserve(
      void** virPtr,
      size_t size,
      size_t alignment,
      void* expectPtr,
      uint64_t flags) = 0;

  // Allocate an address range reservation. e.g. cuMemAddressReserve (without
  // flags)
  virtual int memAddressReserve(
      void** ptr,
      size_t size,
      size_t alignment,
      void* addr) = 0;

  // Create a memory handle representing a memory allocation of a given size
  // described by some default properties. e.g. cuMemCreate, but with some
  // default properties instead.
  virtual int memCreate(
      void** handle,
      size_t size,
      int device,
      uint64_t flags) = 0;

  // Frees the memory that was allocated on a device through memCreate. e.g.
  // cuMemRelease
  virtual int memRelease(void* handle) = 0;

  // Maps an allocation handle to a reserved virtual address range. e.g.
  // cuMemMap
  virtual int memMap(
      void* ptr,
      size_t size,
      size_t offset,
      void* handle,
      uint64_t flags) = 0;

  // Set the access flags for each location for the given virtual address range.
  // e.g. cuMemSetAccess, without desc and count
  virtual int memSetAccess(void* ptr, size_t size, int device) = 0;

  // Unmap the backing memory of a given address range. e.g. cuMemUnmap
  virtual int memUnmap(void* ptr, size_t size) = 0;

  ~CachingAllocatorHelper() = default;
};

// register CachingAllocatorHelper for DefaultCachingAllocator.
void registerHelper(CachingAllocatorHelper* helper);
} // namespace c10::backend::CachingAllocator