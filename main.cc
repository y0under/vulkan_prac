#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>

// TODO: when include vulkan.h, delete the definition
#define VK_MVK_MACOS_SURFACE_EXTENSION_NAME "VK_MVK_macos_surface"

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

/**
 * @brief 
 *
 * @param instance
 * @param pCreateInfo
 * @param pAllocator
 * @param pDebugMessenger
 *
 * @return 
 */
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func =
    (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance,"vkCreateDebugUtilsMessengerEXT");

  if (func != nullptr)
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  else
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

/**
 * @brief 
 *
 * @param instance
 * @param debugMessenger
 * @param pAllocator
 */
void DestroyDebugUtilsMessengerEXT(VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");

  if (func != nullptr)
    func(instance, debugMessenger, pAllocator);
}

/**
 * @brief 
 */
class HelloTriangleApplication {
  public:
    void run() {
      initWindow();
      initVulkan();
      mainLoop();
      cleanup();
    }
  private:
    // struct
    struct QueueFamilyIndices {
      // for drawing commands
      std::optional<uint32_t> graphicsFamily;
      // for presentation
      std::optional<uint32_t> presentFamily;

      bool isComplete()
      {
        return graphicsFamily.has_value() && presentFamily.has_value();
      }
    };

    /**
     * @brief 
     */
    struct SwapChainSupportDetails {
      VkSurfaceCapabilitiesKHR capabilities;
      std::vector<VkSurfaceFormatKHR> formats;
      std::vector<VkPresentModeKHR> presentModes;
    };

    // values
    const uint32_t kwidth = 800;
    const uint32_t kheight = 600;

    const std::vector<const char*> validationLayers = {
      "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char*> deviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    GLFWwindow* window;
    VkInstance instance_;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    // logical device
    VkDevice device_;
    // for queue handle
    VkQueue graphicsQueue;

    VkSurfaceKHR surface_;

    VkSwapchainKHR swap_chain_;
    // for keep VkImage handler
    std::vector<VkImage> swap_chain_images_;
    VkFormat swap_chain_image_format_;
    VkExtent2D swap_chain_extent_;

    std::vector<VkImageView> swap_chain_image_views_;
    // baffer per image
    std::vector<VkFramebuffer> swap_chain_framebuffers_;

    VkRenderPass render_pass_;
    VkPipelineLayout pipeline_layout_;
    VkPipeline graphics_pipeline_;

    VkCommandPool command_pool_;
    VkCommandBuffer command_buffer_;

    // functions

    /**
     * @brief init glfw
     */
    void initWindow() {
      if (!glfwInit()){
        throw std::runtime_error("GLFW not initialized.");
      }
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
      // width, height and title
      window = glfwCreateWindow(kwidth, kheight, "Vulkan test", nullptr, nullptr);
    }

    /**
     * @brief init the app
     */
    void initVulkan() {
      createInstance();
      setupDebugMessenger();
      createSurface();
      pickPhysicalDevice();
      createLogicalDevice();
      createSwapChain();
      createImageViews();
      createRenderPass();
      createGraphicsPipeline();
      createFramebuffers();
      createCommandPool();
      createCommandBuffer();
    }

    /**
     * @brief 
     */
    void createInstance() {
      // if (enableValidationLayers && !checkValidationLayerSupport()) {
      if(enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!!!!11!");
      }

      VkApplicationInfo appInfo{};
      appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      appInfo.pApplicationName = "Hello Triangle";
      appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.pEngineName = "No Engine";
      appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.apiVersion = VK_API_VERSION_1_0;

      VkInstanceCreateInfo create_info{};
      create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      create_info.pApplicationInfo = &appInfo;

      auto extensions = getRequiredExtensions();
      create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
      create_info.ppEnabledExtensionNames = extensions.data();

      // why: for mac code
      create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

      VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
      if (enableValidationLayers) {
        create_info.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        create_info.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
      }
      else {
        create_info.enabledLayerCount = 0;
        // create_info.ppEnabledLayerNames = nullptr;
        create_info.pNext = nullptr;
      }

      // if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
      auto let = vkCreateInstance(&create_info, nullptr, &instance_);
      if (let != VK_SUCCESS) {
        std::cerr << let << std::endl;
        throw std::runtime_error(std::to_string(let) + ": failed to create instance!!!!11!");
      }
    }

    /**
     * @brief 
     */
    void setupDebugMessenger() {
      if (!enableValidationLayers) return;

      VkDebugUtilsMessengerCreateInfoEXT create_info;
      populateDebugMessengerCreateInfo(create_info);

      // if (CreateDebugUtilsMessengerEXT(instance_, &create_info, nullptr, &debugMessenger) !=
      auto let = CreateDebugUtilsMessengerEXT(instance_, &create_info, nullptr, &debugMessenger);
      if (let != VK_SUCCESS) {
        std::cout << let << std::endl;
        throw std::runtime_error(std::to_string(let) + ": failed to set up debug messenger!!!!11!");
      }
    }

    /**
     * @brief create glfw window
     */
    void createSurface() {
      // error case: if vulkan is not supported.
      if (!glfwVulkanSupported()) {
        throw std::runtime_error("Vulkan is not supported!");
      }

      // if (glfwCreateWindowSurface(instance_, window, nullptr, &surface_) != VK_SUCCESS) {
      auto ret = glfwCreateWindowSurface(instance_, window, nullptr, &surface_);
      if (ret != VK_SUCCESS) {
        throw std::runtime_error(std::to_string(ret) + ": failed to create window surface!!!!11!");
      }
    }

    /**
     * @brief select physical device
     */
    void pickPhysicalDevice() {
      uint32_t deviceCount = 0;
      // check devices
      vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
      if (deviceCount == 0)
        throw std::runtime_error("failed to find GPUs with Vulkan support!!!!11!");

      std::vector<VkPhysicalDevice> devices(deviceCount);
      // set devices to vector
      vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

      for (const auto& device: devices) {
        // pick first available device
        if (isDeviceSuitabe(device)) {
          physical_device_ = device;
          break;
        }
      }

      if (physical_device_ == VK_NULL_HANDLE)
        throw std::runtime_error("failed to find a suitable GPU!!!!11!");
    }

    /**
     * @brief 
     */
    void createLogicalDevice() {
      QueueFamilyIndices indices = findQueueFamilies(physical_device_);

      std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};
      std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(),
        indices.presentFamily.value() };

      float queuePriority = 1.0f;
      for (uint32_t queueFamily: uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        queueCreateInfos.emplace_back(queueCreateInfo);
      }

      VkPhysicalDeviceFeatures deviceFeatures{};

      VkDeviceCreateInfo create_info{};
      create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      create_info.pQueueCreateInfos = queueCreateInfos.data();
      create_info.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
      create_info.pEnabledFeatures = &deviceFeatures;
      create_info.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
      create_info.ppEnabledExtensionNames = deviceExtensions.data();

      if (enableValidationLayers) {
        create_info.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        create_info.ppEnabledLayerNames = validationLayers.data();
      }
      else {
        create_info.enabledLayerCount = 0;
      }

      if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!!!!11!");
      }

      // retrive queue handle
      vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue);
    }

    /**
     * @brief 
     */
    void createSwapChain() {
      SwapChainSupportDetails swap_chain_support = querySwapChainSupport(physical_device_);

      VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swap_chain_support.formats);
      VkPresentModeKHR presentMode = chooseSwapPresentMode(swap_chain_support.presentModes);
      VkExtent2D extent = chooseSwapExtent(swap_chain_support.capabilities);

      // why: +1 for performance
      uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
      if (0 < swap_chain_support.capabilities.maxImageCount &&
          swap_chain_support.capabilities.maxImageCount < image_count) {
        image_count = swap_chain_support.capabilities.maxImageCount;
      }

      VkSwapchainCreateInfoKHR create_info{};
      create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
      create_info.surface = surface_;

      create_info.minImageCount = image_count;
      create_info.imageFormat = surfaceFormat.format;
      create_info.imageColorSpace = surfaceFormat.colorSpace;
      create_info.imageExtent = extent;
      // why: 1 means unless developing a stereoscopic 3D application
      create_info.imageArrayLayers = 1;
      create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

      QueueFamilyIndices indices = findQueueFamilies(physical_device_);
      uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

      if (indices.graphicsFamily != indices.presentFamily) {
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = queueFamilyIndices;
      } else {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.queueFamilyIndexCount = 0;
        create_info.pQueueFamilyIndices = nullptr;
      }

      create_info.preTransform = swap_chain_support.capabilities.currentTransform;
      create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
      create_info.presentMode = presentMode;
      create_info.clipped = VK_TRUE;

      // TODO: recreate swapchain
      create_info.oldSwapchain = VK_NULL_HANDLE;

      if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
      }

      vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, nullptr);
      swap_chain_images_.resize(image_count);
      vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, swap_chain_images_.data());

      swap_chain_image_format_ = surfaceFormat.format;
      swap_chain_extent_ = extent;
    }

    /**
     * @brief for determin access way and access area of image
     */
    void createImageViews() {
      swap_chain_image_views_.resize(swap_chain_images_.size());
      for (size_t i = 0; i < swap_chain_images_.size(); ++i) {
        VkImageViewCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        create_info.image = swap_chain_images_[i];
        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format = swap_chain_image_format_;
        create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        create_info.subresourceRange.baseMipLevel = 0;
        create_info.subresourceRange.levelCount = 1;
        create_info.subresourceRange.baseArrayLayer = 0;
        create_info.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device_, &create_info, nullptr, &swap_chain_image_views_[i]) != VK_SUCCESS) {
          throw std::runtime_error("failed to create image views!");
        }
      }
    }

    /**
     * @brief 
     */
    void createRenderPass() {
      VkAttachmentDescription color_attachment{};
      color_attachment.format = swap_chain_image_format_;
      color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;

      color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      // setting for before to start renderpass
      color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      // setting for after end renderpass
      color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

      VkAttachmentReference color_attachment_ref{};
      color_attachment_ref.attachment = 0;
      color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

      VkSubpassDescription subpass{};
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount = 1;
      subpass.pColorAttachments = &color_attachment_ref;

      VkRenderPassCreateInfo render_pass_info{};
      render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      render_pass_info.attachmentCount = 1;
      render_pass_info.pAttachments = &color_attachment;
      render_pass_info.subpassCount = 1;
      render_pass_info.pSubpasses = &subpass;

      if (vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
      }
    }


    /**
     * @brief 
     */
    void createGraphicsPipeline() {
      auto vert_shader_code = ReadFile("shader/vert.spv");
      auto frag_shader_code = ReadFile("shader/frag.spv");

      auto vert_shader_module = createShaderModule(vert_shader_code);
      auto frag_shader_module = createShaderModule(frag_shader_code);

      VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
      vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;

      vert_shader_stage_info.module = vert_shader_module;
      vert_shader_stage_info.pName = "main";

      VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
      frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      frag_shader_stage_info.module = frag_shader_module;
      frag_shader_stage_info.pName = "main";

      VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};

      VkPipelineVertexInputStateCreateInfo vertex_input_info{};
      vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
      vertex_input_info.vertexBindingDescriptionCount = 0;
      vertex_input_info.vertexAttributeDescriptionCount = 0;

      VkPipelineInputAssemblyStateCreateInfo input_assembly{};
      input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      input_assembly.primitiveRestartEnable = VK_FALSE;

      std::vector<VkDynamicState> dynamic_states = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
      };

      VkPipelineDynamicStateCreateInfo dynamic_state{};
      dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
      dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
      dynamic_state.pDynamicStates = dynamic_states.data();

      VkPipelineViewportStateCreateInfo viewport_state{};
      viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      viewport_state.viewportCount = 1;
      // what: for static viewport
      // auto viewport = GenerateViewport();
      // viewport_state.pViewports = &viewport;
      viewport_state.scissorCount = 1;
      // what: for static scissor
      // auto scissor = GenerateScissor();
      // viewport_state.pScissors = &scissor;


      auto rasterizer = GenerateRasterizer();
      auto multisampling = GenerateMultisampling();
      auto color_blend_attachment = GenerateColorBlendAttachment();
      auto color_blending = GenerateColorBlending(color_blend_attachment);
      auto pipeline_layout_info = GeneratePipelineLayoutInfo();

      if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
      }

      VkGraphicsPipelineCreateInfo pipeline_info{};
      pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      pipeline_info.stageCount = 2;
      pipeline_info.pStages = shader_stages;
      pipeline_info.pVertexInputState = &vertex_input_info;
      pipeline_info.pInputAssemblyState = &input_assembly;
      pipeline_info.pViewportState = &viewport_state;
      pipeline_info.pRasterizationState = &rasterizer;
      pipeline_info.pMultisampleState = &multisampling;
      pipeline_info.pDepthStencilState = nullptr;
      pipeline_info.pColorBlendState = &color_blending;
      pipeline_info.pDynamicState = &dynamic_state;
      pipeline_info.layout = pipeline_layout_;
      pipeline_info.renderPass = render_pass_;
      pipeline_info.subpass = 0;
      // for quick changeover pipeline
      pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
      pipeline_info.basePipelineIndex = -1;

      if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
      }

      vkDestroyShaderModule(device_, frag_shader_module, nullptr);
      vkDestroyShaderModule(device_, vert_shader_module, nullptr);
    }

    /**
     * @brief for to use uniform variable in shader
     *
     * @return 
     */
    VkPipelineLayoutCreateInfo GeneratePipelineLayoutInfo() {
      VkPipelineLayoutCreateInfo pipeline_layout_info{};
      pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipeline_layout_info.setLayoutCount = 0;
      pipeline_layout_info.pSetLayouts = nullptr;
      pipeline_layout_info.pushConstantRangeCount = 0;
      pipeline_layout_info.pPushConstantRanges = nullptr;
      return pipeline_layout_info;
    }

    /**
     * @brief setting for global color blending
     *
     * @return 
     */
    VkPipelineColorBlendStateCreateInfo GenerateColorBlending(
        VkPipelineColorBlendAttachmentState &color_blend_attachment) {
      VkPipelineColorBlendStateCreateInfo color_blending{};
      color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      color_blending.logicOpEnable = VK_FALSE;
      color_blending.logicOp = VK_LOGIC_OP_COPY;
      color_blending.attachmentCount = 1;
      color_blending.pAttachments = &color_blend_attachment;
      color_blending.blendConstants[0] = 0.0f;
      color_blending.blendConstants[1] = 0.0f;
      color_blending.blendConstants[2] = 0.0f;
      color_blending.blendConstants[3] = 0.0f;
      return color_blending;
    }

    /**
     * @brief setting for per frame buffer
     *
     * @return 
     */
    VkPipelineColorBlendAttachmentState GenerateColorBlendAttachment() {
      VkPipelineColorBlendAttachmentState color_blend_attachment{};
      color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      color_blend_attachment.blendEnable = VK_FALSE;
      color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
      color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
      return color_blend_attachment;
    }

    /**
     * @brief 
     *
     * @return 
     */
    VkPipelineRasterizationStateCreateInfo GenerateRasterizer() {
      VkPipelineRasterizationStateCreateInfo rasterizer{};
      rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
      rasterizer.depthClampEnable = VK_FALSE;
      rasterizer.rasterizerDiscardEnable = VK_FALSE;
      rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
      rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
      rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
      rasterizer.depthBiasEnable = VK_FALSE;
      rasterizer.depthBiasConstantFactor = 0.0f;
      rasterizer.depthBiasClamp = 0.0f;
      rasterizer.depthBiasSlopeFactor = 0.0f;
      return rasterizer;
    }

    /**
     * @brief for anti-aliasing
     *
     * @return 
     */
    VkPipelineMultisampleStateCreateInfo GenerateMultisampling() {
      VkPipelineMultisampleStateCreateInfo multisampling{};
      multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
      multisampling.sampleShadingEnable = VK_FALSE;
      multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      multisampling.minSampleShading = 1.0f;
      multisampling.pSampleMask = nullptr;
      multisampling.alphaToCoverageEnable = VK_FALSE;
      multisampling.alphaToOneEnable = VK_FALSE;
      return multisampling;
    }

    /**
     * @brief setting for static viewport
     *
     * @return 
     */
    VkViewport GenerateViewport() {
      VkViewport viewport{};
      viewport.x = 0.0f;
      viewport.y = 0.0f;
      viewport.width = (float) swap_chain_extent_.width;
      viewport.height = (float) swap_chain_extent_.height;
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;

      return viewport;
    }

    /**
     * @brief setting for static scissor
     */
    VkRect2D GenerateScissor() {
      VkRect2D scissor{};
      scissor.offset = {0, 0};
      scissor.extent = swap_chain_extent_;
      return scissor;
    }

    /**
     * @brief 
     *
     * @param code
     *
     * @return 
     */
    VkShaderModule createShaderModule(const std::vector<char> &code) {
      VkShaderModuleCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      createInfo.codeSize = code.size();
      createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

      VkShaderModule shaderModule;
      if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
      }

      return shaderModule;
    }

    /**
     * @brief 
     */
    void createFramebuffers() {
      swap_chain_framebuffers_.resize(swap_chain_image_views_.size());

      for (size_t i = 0; i < swap_chain_image_views_.size(); ++i) {
        VkImageView attachments[] = {
          swap_chain_image_views_[i]
        };

        VkFramebufferCreateInfo framebuffer_info{};
        framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_info.renderPass = render_pass_;
        framebuffer_info.attachmentCount = 1;
        framebuffer_info.pAttachments = attachments;
        framebuffer_info.width = swap_chain_extent_.width;
        framebuffer_info.height = swap_chain_extent_.height;
        framebuffer_info.layers = 1;

        if (vkCreateFramebuffer(device_, &framebuffer_info, nullptr, &swap_chain_framebuffers_[i]) != VK_SUCCESS) {
          throw std::runtime_error("failed to create framebuffer!");
        }
      }
    }

    /**
     * @brief 
     */
    void createCommandPool() {
      QueueFamilyIndices queue_family_indices = findQueueFamilies(physical_device_);

      VkCommandPoolCreateInfo pool_info{};
      pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      pool_info.queueFamilyIndex = queue_family_indices.graphicsFamily.value();

      if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
      }
    }

    /**
     * @brief 
     */
    void createCommandBuffer() {
      VkCommandBufferAllocateInfo alloc_info{};
      alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      alloc_info.commandPool = command_pool_;
      // primary or secondery
      alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      alloc_info.commandBufferCount = 1;

      if (vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer_) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
      }
    }

    /**
     * @brief write command to command buffer
     *
     * @param command_buffer
     * @param image_index
     */
    void recordCommandBuffer(VkCommandBuffer command_buffer, uint32_t image_index) {
      VkCommandBufferBeginInfo begin_info{};
      begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin_info.flags = 0;
      begin_info.pInheritanceInfo = nullptr;

      if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
      }

      // to start renderpass
      VkRenderPassBeginInfo render_pass_info{};
      render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      render_pass_info.renderPass = render_pass_;
      render_pass_info.framebuffer = swap_chain_framebuffers_[image_index];

      render_pass_info.renderArea.offset = {0, 0};
      render_pass_info.renderArea.extent = swap_chain_extent_;

      VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
      render_pass_info.clearValueCount = 1;
      render_pass_info.pClearValues = &clear_color;

      vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);

      auto viewport = GenerateViewport();
      vkCmdSetViewport(command_buffer, 0, 1, &viewport);
      auto scissor = GenerateScissor();
      vkCmdSetScissor(command_buffer, 0, 1, &scissor);

      vkCmdDraw(command_buffer, 3, 1, 0, 0);

      vkCmdEndRenderPass(command_buffer);
      if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }

    /**
     * @brief quering details of swap chain support
     *
     * @param device
     *
     * @return 
     */
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
      SwapChainSupportDetails details;
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

      uint32_t formatCount;
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, nullptr);

      if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, details.formats.data());
      }

      uint32_t presentModeCount;
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, nullptr);

      if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, details.presentModes.data());
      }

      return details;
    }

    /**
     * @brief 
     *
     * @param availableFormats
     *
     * @return 
     */
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
      for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
          return availableFormat;
        }
      }

      return availableFormats[0];
    }

    /**
     * @brief choose available and most useful
     *
     * @param availablePresentModes
     *
     * @return 
     */
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) {
      for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
          return availablePresentMode;
        }
      }
      return VK_PRESENT_MODE_FIFO_KHR;

    }

    /**
     * @brief 
     *
     * @param capabilities
     *
     * @return 
     */
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
      if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
      } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
      }
    }

    /**
     * @brief 
     *
     * @param device
     *
     * @return 
     */
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
      uint32_t queueFamilyCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
      std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

      QueueFamilyIndices indices;

      int i = 0;
      for (const auto& queueFamily: queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
          indices.graphicsFamily = i;

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);

        if (presentSupport)
          indices.presentFamily = i;

        if (indices.isComplete())
          break;

        ++i;
      }
      return indices;
    }

    /**
     * @brief state of device suitability
     *
     * @param device
     *
     * @return 
     */
    bool isDeviceSuitabe(VkPhysicalDevice device) {
      QueueFamilyIndices indices = findQueueFamilies(device);

      bool extensionsSupported = checkDeviceExtensionSupport(device);

      // for swap chain support
      bool swapChainAdequate = false;
      if (extensionsSupported) {
        SwapChainSupportDetails swap_chain_support = querySwapChainSupport(device);
        swapChainAdequate = !swap_chain_support.formats.empty() && !swap_chain_support.presentModes.empty();
      }

      return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    /**
     * @brief 
     *
     * @param device
     *
     * @return 
     */
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
      uint32_t extensionCount;
      vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

      std::vector<VkExtensionProperties> availableExtensions(extensionCount);
      vkEnumerateDeviceExtensionProperties(
          device, nullptr, &extensionCount, availableExtensions.data());

      std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

      for (const auto& extension: availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
      }

      return requiredExtensions.empty();
    }


    /**
     * @brief 
     */
    void mainLoop() {
      while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE)) break;
        glfwPollEvents();
        DrawFrame();
      }
    }


    /**
     * @brief 
     */
    void DrawFrame() {

    }

    /**
     * @brief destruct the app
     */
    void cleanup() {
      vkDestroyCommandPool(device_, command_pool_, nullptr);
      for (auto buffer : swap_chain_framebuffers_) {
        vkDestroyFramebuffer(device_, buffer, nullptr);
      }
      vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
      vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
      vkDestroyRenderPass(device_, render_pass_, nullptr);
      vkDestroySwapchainKHR(device_, swap_chain_, nullptr);
      vkDestroyDevice(device_, nullptr);

      if (enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(instance_, debugMessenger, nullptr);
      }

      vkDestroySurfaceKHR(instance_, surface_, nullptr);
      vkDestroyInstance(instance_, nullptr);
      glfwDestroyWindow(window);
      glfwTerminate();
    }

    /**
     * @brief 
     *
     * @return 
     */
    bool checkValidationLayerSupport() {
      uint32_t layerCount;
      vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

      std::vector<VkLayerProperties> availableLayers(layerCount);
      vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

      for (const char* layerName: validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties: availableLayers) {
          if (strcmp(layerName, layerProperties.layerName) == 0) {
            layerFound = true;
            break;
          }
        }

        if (!layerFound) {
          std::cout << "Line: " << __LINE__ <<  " return false" << std::endl;
          return false;
        }
      }

      return true;
    }

    /**
     * @brief 
     *
     * @param create_info
     */
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
      create_info = {};
      create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

      create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

      create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

      create_info.pfnUserCallback = debugCallback;
      create_info.pUserData = nullptr;
    }

    /**
     * @brief 
     *
     * @return 
     */
    std::vector<const char*> getRequiredExtensions() {
      uint32_t glfwExtensionCount = 0;
      const char** glfwExtensions;
      glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

      std::cout << "glfwExtensionCount: " << glfwExtensionCount << "\n";

      std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

      // why: for mac code
      extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
      extensions.emplace_back(VK_KHR_SURFACE_EXTENSION_NAME);
      extensions.emplace_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);

      if (enableValidationLayers) {
        extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      }

      return extensions;
    }

    /**
     * @brief 
     *
     * @param messageSeverity
     * @param messageType
     * @param pCallbackData
     * @param pUserData
     *
     * @return 
     */
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {
      std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

      return VK_FALSE;
    }

    /**
     * @brief for load shader
     *
     * @param file_name
     *
     * @return 
     */
    static std::vector<char> ReadFile(const std::string &file_name) {
      std::ifstream file(file_name, std::ios::ate | std::ios::binary);

      if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
      }

      size_t fileSize = (size_t) file.tellg();
      std::vector<char> buffer(fileSize);
      file.seekg(0);
      file.read(buffer.data(), fileSize);
      file.close();

      return buffer;
    }
};

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
