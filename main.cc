#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

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

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

    GLFWwindow* window;
    VkInstance instance;

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
    }

    void createInstance()
    {
      if (enableValidationLayers && !checkValidationLayerSupport()) {
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
      createInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      createInfo.pApplicationInfo = &appInfo;

      uint32_t glfwExtensionCount = 0;
      const char** glfwExtensions;

      glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

      std::vector<const char*> requiredGlfwExtensions;

      for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
        requiredGlfwExtensions.emplace_back(glfwExtensions[i]);
      }

      // why: the code is defferent between tutrial code
      //      because got VK_ERROR_INCOMPATIBLE_DRIVER with vkCreateInstance execution. 
      requiredGlfwExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

      createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

      createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredGlfwExtensions.size());
      createInfo.ppEnabledExtensionNames = requiredGlfwExtensions.data();

      createInfo.enabledLayerCount = 0;

      if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!!!!11!");
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
      vkDestroyInstance(instance, nullptr);
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

        if (!layerFound)
          return false;
      }

      return true;
    }
};

int main()
{
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
