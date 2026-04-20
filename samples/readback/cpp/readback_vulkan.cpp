//
// Created by ByteDance on 12/17/25.
//
#ifdef XR_USE_GRAPHICS_API_VULKAN
#include "readback_file.h"
#include <assert.h>

extern VkDevice g_device;
extern VkPhysicalDevice g_pdevice;
namespace SecureMR {
  void ReadbackCheck::initializeGraphicsContext() {
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pdevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(g_pdevice, &queueFamilyCount, families.data());

    int graphicsFamily = -1;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
      const auto& f = families[i];
      if (f.queueFlags & VK_QUEUE_GRAPHICS_BIT) graphicsFamily = i;
    }
    int queueFamilyIndex = graphicsFamily;
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = queueFamilyIndex;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(g_device, &ci, nullptr, &cmdPool);
    vkGetDeviceQueue(g_device, graphicsFamily, 0, &queue);
    assert(queue != VK_NULL_HANDLE);
  }

  void ReadbackCheck::OutputVulkanTextureToPath(XrReadbackTextureImageVulkanPICO * vTexture, const std::string &path) {
      VkImage srcImage = vTexture->image;
      uint32_t w = mConfig.w, h = mConfig.h;

      VkDeviceSize imageSize = (VkDeviceSize)w * h * 4;
      VkBufferCreateInfo bci = {};
      bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bci.size = imageSize;
      bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      VkBuffer stagingBuf;
      vkCreateBuffer(g_device, &bci, NULL, &stagingBuf);

      VkMemoryRequirements mr;
      vkGetBufferMemoryRequirements(g_device, stagingBuf, &mr);

      VkPhysicalDeviceMemoryProperties memProps;
      vkGetPhysicalDeviceMemoryProperties(g_pdevice, &memProps);
      uint32_t memType;
      auto desiredProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        bool supported = mr.memoryTypeBits & (1 << i);
        bool hasProps = (memProps.memoryTypes[i].propertyFlags & desiredProps) == desiredProps;
        if (supported && hasProps) {
          memType = i;
          break;
        }
      }

      VkMemoryAllocateInfo mai = {};
      mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      mai.allocationSize = mr.size;
      mai.memoryTypeIndex = memType;

      VkDeviceMemory stagingMem;
      vkAllocateMemory(g_device, &mai, NULL, &stagingMem);
      vkBindBufferMemory(g_device, stagingBuf, stagingMem, 0);
      auto beginOneTime = [&](VkCommandBuffer* out) {
        VkCommandBufferAllocateInfo a = {};
        a.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        a.commandPool = cmdPool;
        a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        a.commandBufferCount = 1;
        vkAllocateCommandBuffers(g_device, &a, out);

        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(*out, &bi);
      };
      auto endOneTime = [&](VkCommandBuffer cb) {
        vkEndCommandBuffer(cb);
        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(g_device, cmdPool, 1, &cb);
      };

      VkCommandBuffer cb;
      beginOneTime(&cb);

      VkImageMemoryBarrier toSrc{};
      toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      toSrc.pNext = nullptr;

      toSrc.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

      toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

      toSrc.image = srcImage;

      toSrc.subresourceRange = {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = VK_REMAINING_MIP_LEVELS,
          .baseArrayLayer = 0,
          .layerCount = VK_REMAINING_ARRAY_LAYERS,
      };

      toSrc.srcAccessMask = 0;
      toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

      vkCmdPipelineBarrier(
          cb,
          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          0,
          0, nullptr,
          0, nullptr,
          1, &toSrc
      );
      VkBufferImageCopy region = {0};
      region.bufferOffset = 0;
      region.bufferRowLength = 0;
      region.bufferImageHeight = 0;
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.mipLevel = 0;
      region.imageSubresource.baseArrayLayer = 0;
      region.imageSubresource.layerCount = 1;
      region.imageOffset = (VkOffset3D){0, 0, 0};
      region.imageExtent = (VkExtent3D){w, h, 1};

      vkCmdCopyImageToBuffer(cb, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuf, 1, &region);

      VkBufferMemoryBarrier bufBarrier = {};
      bufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      bufBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      bufBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bufBarrier.buffer = stagingBuf;
      bufBarrier.offset = 0;
      bufBarrier.size = imageSize;

      vkCmdPipelineBarrier(
          cb,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_HOST_BIT,
          0, 0, nullptr, 1, &bufBarrier, 0, nullptr);

      endOneTime(cb);
      void* mapped = nullptr;
      vkMapMemory(g_device, stagingMem, 0, imageSize, 0, &mapped);

      bool flipY = false;
      unsigned char* pixels = (unsigned char*)mapped;
      if (flipY) {
        size_t stride = (size_t)w * 4;
        for (uint32_t y = 0; y < h / 2; ++y) {
          unsigned char* rowA = pixels + y * stride;
          unsigned char* rowB = pixels + (h - 1 - y) * stride;
          for (size_t i = 0; i < stride; ++i) {
            unsigned char t = rowA[i];
            rowA[i] = rowB[i];
            rowB[i] = t;
          }
        }
      }
      auto ret = stbi_write_png(path.c_str(), (int)w, (int)h, 3, pixels, (int)w * 3);
      if (ret == 0) {
        LOGE("readback vulkan texture failed: %d", ret);
      }
      vkUnmapMemory(g_device, stagingMem);
  }

  bool ReadbackCheck::OutputReadbackTextureToPath(const XrReadbackTexturePICO& texture, const std::string& path) {
    XrReadbackTextureImageVulkanPICO vulkan_texture;
    auto ret = mReadbackController->RetrieveTexture(texture, (XrReadbackTextureImageBaseHeaderPICO*)&vulkan_texture);
    if (!ret) {
      return false;
    }
    OutputVulkanTextureToPath(&vulkan_texture, path);
    return true;
  }

}
#endif