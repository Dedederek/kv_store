#https://github.com/NVIDIA/cuda-python/blob/main/examples/0_Introduction/vectorAddMMAP_test.py

from cuda import cuda
import time
import torch
import cupy

get_start_addr_from_kv_id = {}
get_max_len_from_kv_id = {}
get_map_len_from_kv_id = {}
get_mem_handle_list_from_kv_id = {}

request_kv_id = 0
curr_len = 0
leftover_addr_space = 0
granularity = 2097152

def round_up(x, y):
    return int((x - 1)/y + 1) * y

def increase_kv_size(kv_id, incremental_size): # return start address, valid address space: return_addr+incremental_size
    global granularity, curr_len, leftover_addr_space
    return_addr = get_start_addr_from_kv_id[kv_id] + curr_len
    if incremental_size <= leftover_addr_space:
        curr_len = curr_len + incremental_size
        leftover_addr_space = leftover_addr_space - incremental_size
    else:
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        remain_incremental_size = incremental_size - leftover_addr_space
        size = round_up(remain_incremental_size, granularity)
        status, allocationHandle = cuda.cuMemCreate(size, prop, 0)
        if status != cuda.CUresult.CUDA_SUCCESS:
            print("cuMemCreate failed")
        status, = cuda.cuMemMap(get_start_addr_from_kv_id[kv_id] + curr_len + leftover_addr_space, size, 0, allocationHandle, 0)

        if status != cuda.CUresult.CUDA_SUCCESS:
            print("cuMemMap failed")

        if kv_id not in get_map_len_from_kv_id:
            get_map_len_from_kv_id[kv_id] = [size]
        else:
            get_map_len_from_kv_id[kv_id].append(size)

        if kv_id not in get_mem_handle_list_from_kv_id:
            get_mem_handle_list_from_kv_id[kv_id] = [allocationHandle]
        else:
            get_mem_handle_list_from_kv_id[kv_id].append(allocationHandle)

        accessDescriptors = cuda.CUmemAccessDesc()
        accessDescriptors.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        accessDescriptors.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        status, = cuda.cuMemSetAccess(get_start_addr_from_kv_id[kv_id]+ curr_len + leftover_addr_space, size, [accessDescriptors], 1)
        if status != cuda.CUresult.CUDA_SUCCESS:
            print("cuMemSetAccess failed")

        curr_len = curr_len + leftover_addr_space + remain_incremental_size
        leftover_addr_space = size - remain_incremental_size

    return return_addr

def create_kv(device_id, max_reserve_len, incre_size): #reserve virtual memory space with max_reserve_len size
    global request_kv_id
    request_kv_id = request_kv_id + 1
    cuda.cuInit(device_id)
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    global granularity
    status, granularity = cuda.cuMemGetAllocationGranularity(prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)
    print("granularity is ",granularity)
    if status != cuda.CUresult.CUDA_SUCCESS:
        print("get allocation granularity failed")
    size = round_up(max_reserve_len, granularity) # 2097152 = 2^21 = 2MB
    
     #desired virtual memory addr
    status, dptr = cuda.cuMemAddressReserve(size, 0, None, 0)
    get_start_addr_from_kv_id[request_kv_id] = int(dptr)
    print("get_start_addr_from_kv_id is ", hex(get_start_addr_from_kv_id[request_kv_id]))
    get_max_len_from_kv_id[request_kv_id] = size
    #increase_kv_size(request_kv_id, incre_size)
    return request_kv_id

def get_kv_addr(kv_id): ## start addr or addr offset?
    #print("get kv addr", request_addr_map[id])
    return get_start_addr_from_kv_id[kv_id]

def destory_kv(kv_id):
    for i in range(len(get_map_len_from_kv_id[kv_id])):
        if i==0:
            status = cuda.cuMemUnmap(get_start_addr_from_kv_id[kv_id], get_map_len_from_kv_id[kv_id][i])
        else:
            status = cuda.cuMemUnmap(get_start_addr_from_kv_id[kv_id] + get_map_len_from_kv_id[kv_id][i-1], get_map_len_from_kv_id[kv_id][i])
        if status[0] != cuda.CUresult.CUDA_SUCCESS:
            print("cuMemUnmap failed: ", status)
        status = cuda.cuMemRelease(get_mem_handle_list_from_kv_id[kv_id][i])
        if status[0] != cuda.CUresult.CUDA_SUCCESS:
            print("cuMemRelease failed")
   
    status = cuda.cuMemAddressFree(get_start_addr_from_kv_id[kv_id], get_max_len_from_kv_id[kv_id])
    if status[0] != cuda.CUresult.CUDA_SUCCESS:
        print("cuMemAddressFree failed:", status[0])

# convert raw device memory to torch tensor
def ptr_to_tensor(device_ptr: int, size: int, shape: tuple):
    mem = cupy.cuda.UnownedMemory(device_ptr, size, None, 0)
    memptr = cupy.cuda.MemoryPointer(mem, 0)
    arr = cupy.ndarray(shape, dtype=cupy.float32, memptr=memptr)
    return torch.as_tensor(arr, device="cuda")

cuda.cuInit(0)
id = create_kv(0,200000000000,1)
addr = increase_kv_size(id, 300)
torch_tensor = ptr_to_tensor(addr, 300, (300,))
addr = increase_kv_size(id, 5097153)
destory_kv(id)
