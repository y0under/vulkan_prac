#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <optional>

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger)
{
  auto func =
    (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance,"vkCreateDebugUtilsMessengerEXT");

  if (func != nullptr)
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  else
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}


void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator)
{
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");

  if (func != nullptr)
    func(instance, debugMessenger, pAllocator);
}

class HelloTriangleApplication
{
  public:
    void run()
    {
      initWindow();
      initVulkan();
      mainLoop();
      cleanup();
    }
  private:
    // values
    const uint32_t kwidth = 800;
    const uint32_t kheight = 600;

    const std::vector<const char*> validationLayers = {
      "VK_LAYER_KHRONOS_validation"
    };

    GLFWwindow* window;
    VkInstance instance_;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;

    // functions
    void initWindow()
    {
      glfwInit();
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
      // width, height and title
      window = glfwCreateWindow(kwidth, kheight, "Vulkan", nullptr, nullptr);
    }

    void initVulkan()
    {
      createInstance();
      setupDebugMessenger();
      pickPhysicalDevice();
    }

    /*
     * select physical device
     */
    void pickPhysicalDevice()
    {
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
          physicalDevice_ = device;
          break;
        }
      }

      if (physicalDevice_ == VK_NULL_HANDLE)
        throw std::runtime_error("failed to find a suitable GPU!!!!11!");
    }

    struct QueueFamilyIndices
    {
      std::optional<uint32_t> graphicsFamily;

      bool isComplete()
      {
        return graphicsFamily.has_value();
      }
    };

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
    {
      uint32_t queueFamilyCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
      std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

      QueueFamilyIndices indices;

      int i = 0;
      for (const auto& queueFamily: queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
          indices.graphicsFamily = i;
        if (indices.isComplete())
          break;
        ++i;
      }
      return indices;
    }

    bool isDeviceSuitabe(VkPhysicalDevice device)
    {
      QueueFamilyIndices indices = findQueueFamilies(device);

      return indices.isComplete();
    }

    void createInstance()
    {
      // if (enableValidationLayers && !checkValidationLayerSupport()) {
      auto is_available_validation = enableValidationLayers && !checkValidationLayerSupport();
      if (is_available_validation){
        throw std::runtime_error("validation layers requested, but not available!!!!11!");
      }

      VkApplicationInfo appInfo{};
      appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      appInfo.pApplicationName = "Hello Triangle";
      appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.pEngineName = "No Engine";
      appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.apiVersion = VK_API_VERSION_1_0;

      VkInstanceCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      createInfo.pApplicationInfo = &appInfo;

      auto extensions = getRequiredExtensions();
      createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
      createInfo.ppEnabledExtensionNames = extensions.data();

      // why: for mac code
      createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

      /*
      createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredGlfwExtensions.size());
      createInfo.ppEnabledExtensionNames = requiredGlfwExtensions.data();
      */

      VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
      if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
      }
      else {
        createInfo.enabledLayerCount = 0;
        createInfo.ppEnabledLayerNames = nullptr;
      }

      // if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
      auto let = vkCreateInstance(&createInfo, nullptr, &instance_);
        if (let != VK_SUCCESS) {
          std::cerr << let << std::endl;
        throw std::runtime_error(std::to_string(let) + ": failed to create instance!!!!11!");
        }
    }

    void mainLoop()
    {
      while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
      }
    }

    void cleanup()
    {
      if (enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(instance_, debugMessenger, nullptr);
      }

      vkDestroyInstance(instance_, nullptr);
      glfwDestroyWindow(window);
      glfwTerminate();
    }

    bool checkValidationLayerSupport()
    {
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
          std::cout << "return false" << std::endl;
          return false;
        }
      }

      return true;
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
      createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

      createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                   VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

      createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

      createInfo.pfnUserCallback = debugCallback;
      createInfo.pUserData = nullptr;
    }

    void setupDebugMessenger()
    {
      if (!enableValidationLayers) return;

      VkDebugUtilsMessengerCreateInfoEXT createInfo;
      populateDebugMessengerCreateInfo(createInfo);

      // if (CreateDebugUtilsMessengerEXT(instance_, &createInfo, nullptr, &debugMessenger) !=
      auto let = CreateDebugUtilsMessengerEXT(instance_, &createInfo, nullptr, &debugMessenger);
          if (let != VK_SUCCESS) {
            std::cout << let << std::endl;
        throw std::runtime_error(std::to_string(let) + ": failed to set up debug messenger!!!!11!");
      }
    }

   std::vector<const char*> getRequiredExtensions()
    {
      uint32_t glfwExtensionCount = 0;
      const char** glfwExtensions;
      glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

      std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

      // why: for mac code
      extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

      if (enableValidationLayers) {
        extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      }

      return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
      std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

      return VK_FALSE;
    }

};

int main()
{
  HelloTriangleApplication app;

  std::cout << (enableValidationLayers ? "true" : "false") << std::endl;

  try {
    app.run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
