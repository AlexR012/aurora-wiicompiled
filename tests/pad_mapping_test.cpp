#include <gtest/gtest.h>

#include "gamepad.hpp"
#include "internal.hpp"
#include "io.hpp"

#include <SDL3/SDL_init.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_timer.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace {

class PadMappingTest : public testing::Test {
protected:
  void SetUp() override {
    savedControllers = std::move(aurora::gamepad::g_GameControllers);
    aurora::gamepad::g_GameControllers.clear();
    savedPath = aurora::g_config.userPath;
    initialized = SDL_InitSubSystem(SDL_INIT_GAMEPAD);
    ASSERT_TRUE(initialized) << SDL_GetError();
    SDL_VirtualJoystickDesc probe{};
    SDL_INIT_INTERFACE(&probe);
    const auto id = SDL_AttachVirtualJoystick(&probe);
    if (id == 0 && std::string_view(SDL_GetError()) == "SDL not built with virtual-joystick support") {
      GTEST_SKIP() << SDL_GetError();
    }
    ASSERT_NE(id, 0u) << SDL_GetError();
    ASSERT_TRUE(SDL_DetachVirtualJoystick(id)) << SDL_GetError();
    for (unsigned int attempt = 0; directory.empty(); ++attempt) {
      const auto candidate = std::filesystem::temp_directory_path() /
          fmt::format("aurora-pad-mapping-{}-{}", SDL_GetTicksNS(), attempt);
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error)) {
        directory = candidate;
      }
      ASSERT_FALSE(error) << error.message();
    }
    path = aurora::io::fs_path_to_string(directory);
    aurora::g_config.userPath = path.c_str();
  }

  void TearDown() override {
    for (const auto& [id, controller] : aurora::gamepad::g_GameControllers) {
      SDL_CloseGamepad(controller.m_controller);
    }
    aurora::gamepad::g_GameControllers = std::move(savedControllers);
    for (const auto id : devices) {
      SDL_DetachVirtualJoystick(id);
    }
    aurora::g_config.userPath = savedPath;
    std::error_code error;
    if (!directory.empty()) {
      std::filesystem::remove_all(directory, error);
    }
    if (initialized) {
      SDL_QuitSubSystem(SDL_INIT_GAMEPAD);
    }
  }

  SDL_JoystickID add(int port, bool adapter = false, uint16_t product = 0x5678) {
    SDL_VirtualJoystickDesc desc{};
    SDL_INIT_INTERFACE(&desc);
    desc.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    desc.naxes = SDL_GAMEPAD_AXIS_COUNT;
    desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    desc.axis_mask = (1u << SDL_GAMEPAD_AXIS_COUNT) - 1;
    desc.button_mask = (1u << SDL_GAMEPAD_BUTTON_COUNT) - 1;
    desc.vendor_id = adapter ? 0x057e : 0x1234;
    desc.product_id = adapter ? 0x0337 : product;
    desc.name = "mapping test pad";
    const auto id = SDL_AttachVirtualJoystick(&desc);
    if (id == 0) {
      ADD_FAILURE() << SDL_GetError();
      return 0;
    }
    devices.push_back(id);
    auto* gamepad = SDL_OpenGamepad(id);
    if (gamepad == nullptr) {
      ADD_FAILURE() << SDL_GetError();
      return 0;
    }
    aurora::gamepad::GameController controller;
    controller.m_controller = gamepad;
    controller.m_index = id;
    controller.m_isGameCube = adapter;
    controller.m_vid = desc.vendor_id;
    controller.m_pid = desc.product_id;
    aurora::gamepad::g_GameControllers[id] = controller;
    aurora::gamepad::set_player_index(id, port);
    return id;
  }

  void remove(SDL_JoystickID id) {
    auto& controllers = aurora::gamepad::g_GameControllers;
    SDL_CloseGamepad(controllers.at(id).m_controller);
    controllers.erase(id);
  }

  void bind(u32 port, SDL_GamepadButton button) {
    u32 count = 0;
    ASSERT_NE(PADGetButtonMappings(port, &count), nullptr);
    ASSERT_EQ(count, PAD_BUTTON_COUNT);
    PADSetButtonMapping(port, {static_cast<u32>(button), PAD_BUTTON_A});
  }

  std::vector<uint8_t> contents(SDL_JoystickID id) {
    const auto& controller = aurora::gamepad::g_GameControllers.at(id);
    const auto file = directory / fmt::format("{}_{:04X}_{:04X}.controller",
                                            aurora::gamepad::controller_name(id), controller.m_vid, controller.m_pid);
    const auto bytes = aurora::io::read_file(file);
    EXPECT_TRUE(bytes.has_value()) << SDL_GetError();
    return bytes.value_or(std::vector<uint8_t>{});
  }

  void expect_binding(u32 port, SDL_GamepadButton button) {
    u32 count = 0;
    const auto* mappings = PADGetButtonMappings(port, &count);
    ASSERT_NE(mappings, nullptr);
    ASSERT_EQ(count, PAD_BUTTON_COUNT);
    const auto mapping = std::find_if(mappings, mappings + count,
                                      [](const auto& value) { return value.padButton == PAD_BUTTON_A; });
    ASSERT_NE(mapping, mappings + count);
    EXPECT_EQ(mapping->nativeButton, static_cast<u32>(button));
  }

  decltype(aurora::gamepad::g_GameControllers) savedControllers;
  const char* savedPath = nullptr;
  bool initialized = false;
  std::filesystem::path directory;
  std::string path;
  std::vector<SDL_JoystickID> devices;
};

TEST_F(PadMappingTest, NeverAssignedPadDoesNotCreateAMappingFile) {
  const auto id = add(-1);
  ASSERT_NE(id, 0u);
  PADSerializeMappings();
  EXPECT_FALSE(aurora::gamepad::g_GameControllers.at(id).m_mappingLoaded);
  for (const auto& file : std::filesystem::directory_iterator(directory)) {
    EXPECT_NE(file.path().extension(), ".controller");
  }
}

TEST_F(PadMappingTest, NeverAssignedSameModelPadsPreserveSavedBindings) {
  const auto original = add(0);
  ASSERT_NE(original, 0u);
  bind(0, SDL_GAMEPAD_BUTTON_NORTH);
  PADSerializeMappings();
  const auto saved = contents(original);
  ASSERT_FALSE(saved.empty());
  remove(original);

  const auto extra = add(-1);
  ASSERT_NE(extra, 0u);
  ASSERT_NE(add(-1), 0u);
  const auto assigned = add(0, false, 0x5679);
  ASSERT_NE(assigned, 0u);
  bind(0, SDL_GAMEPAD_BUTTON_WEST);
  PADSerializeMappings();
  EXPECT_FALSE(aurora::gamepad::g_GameControllers.at(extra).m_mappingLoaded);
  EXPECT_EQ(contents(extra), saved);
  ASSERT_FALSE(contents(assigned).empty());
  remove(assigned);
  ASSERT_NE(add(0, false, 0x5679), 0u);
  expect_binding(0, SDL_GAMEPAD_BUTTON_WEST);

  aurora::gamepad::set_player_index(extra, 1);
  expect_binding(1, SDL_GAMEPAD_BUTTON_NORTH);
}

TEST_F(PadMappingTest, LoadedBindingsSurviveNoneAndReconnect) {
  const auto original = add(0);
  ASSERT_NE(original, 0u);
  bind(0, SDL_GAMEPAD_BUTTON_WEST);
  aurora::gamepad::set_player_index(original, -1);
  PADSerializeMappings();
  const auto saved = contents(original);
  ASSERT_FALSE(saved.empty());
  remove(original);

  const auto reconnected = add(-1);
  ASSERT_NE(reconnected, 0u);
  PADSerializeMappings();
  EXPECT_EQ(contents(reconnected), saved);
  aurora::gamepad::set_player_index(reconnected, 0);
  expect_binding(0, SDL_GAMEPAD_BUTTON_WEST);
}

TEST_F(PadMappingTest, UnsavedEditSurvivesRemapAndUnassignment) {
  const auto id = add(0);
  ASSERT_NE(id, 0u);
  bind(0, SDL_GAMEPAD_BUTTON_NORTH);
  PADSerializeMappings();
  bind(0, SDL_GAMEPAD_BUTTON_WEST);
  ASSERT_TRUE(aurora::gamepad::refresh_controller(id));
  aurora::gamepad::set_player_index(id, -1);
  PADSerializeMappings();
  remove(id);
  ASSERT_NE(add(0), 0u);
  expect_binding(0, SDL_GAMEPAD_BUTTON_WEST);
}

TEST_F(PadMappingTest, ReassignedPadWinsSharedFileOnRepeatedSaves) {
  const auto oldPad = add(0);
  ASSERT_NE(oldPad, 0u);
  bind(0, SDL_GAMEPAD_BUTTON_NORTH);
  PADSerializeMappings();
  bind(0, SDL_GAMEPAD_BUTTON_EAST);
  aurora::gamepad::set_player_index(oldPad, -1);
  const auto newPad = add(0);
  ASSERT_NE(newPad, 0u);
  bind(0, SDL_GAMEPAD_BUTTON_WEST);
  PADSerializeMappings();
  PADSerializeMappings();
  remove(oldPad);
  remove(newPad);
  ASSERT_NE(add(0), 0u);
  expect_binding(0, SDL_GAMEPAD_BUTTON_WEST);
}

TEST_F(PadMappingTest, IncompleteMappingIsPreserved) {
  const auto id = add(0);
  ASSERT_NE(id, 0u);
  bind(0, SDL_GAMEPAD_BUTTON_WEST);
  PADSerializeMappings();
  auto bytes = contents(id);
  ASSERT_GT(bytes.size(), 32u);
  bytes.resize(33);
  const auto file = directory / fmt::format("{}_1234_5678.controller", aurora::gamepad::controller_name(id));
  ASSERT_TRUE(aurora::io::write_file(file, bytes));
  remove(id);
  const auto replacement = add(0);
  ASSERT_NE(replacement, 0u);
  PADSerializeMappings();
  EXPECT_EQ(contents(replacement), bytes);
  bind(0, SDL_GAMEPAD_BUTTON_EAST);
  PADSerializeMappings();
  remove(replacement);
  ASSERT_NE(add(0), 0u);
  expect_binding(0, SDL_GAMEPAD_BUTTON_EAST);
}

TEST_F(PadMappingTest, FailedPublishRetainsTheAssignedEditForRetry) {
  const auto oldPad = add(0);
  ASSERT_NE(oldPad, 0u);
  bind(0, SDL_GAMEPAD_BUTTON_NORTH);
  PADSerializeMappings();
  bind(0, SDL_GAMEPAD_BUTTON_EAST);
  aurora::gamepad::set_player_index(oldPad, -1);
  const auto newPad = add(0);
  ASSERT_NE(newPad, 0u);
  bind(0, SDL_GAMEPAD_BUTTON_WEST);
  const auto file = directory / fmt::format("{}_1234_5678.controller", aurora::gamepad::controller_name(newPad));
  const auto backup = directory / "saved";
  std::filesystem::rename(file, backup);
  ASSERT_TRUE(std::filesystem::create_directories(file / "occupied"));
  PADSerializeMappings();
  ASSERT_TRUE(std::filesystem::is_directory(file / "occupied"));
  std::filesystem::remove_all(file);
  std::filesystem::rename(backup, file);
  PADSerializeMappings();
  remove(oldPad);
  remove(newPad);
  ASSERT_NE(add(0), 0u);
  expect_binding(0, SDL_GAMEPAD_BUTTON_WEST);
}

TEST_F(PadMappingTest, FirstDeadZoneEditReplacesAnInvalidStandardMapping) {
  const auto id = add(0);
  ASSERT_NE(id, 0u);
  const auto file = directory / fmt::format("{}_1234_5678.controller", aurora::gamepad::controller_name(id));
  const std::vector<uint8_t> invalid(4096, 0xff);
  ASSERT_TRUE(aurora::io::write_file(file, invalid));
  auto* zones = PADGetDeadZones(0);
  ASSERT_NE(zones, nullptr);
  zones->stickDeadZone = 5432;
  PADSerializeMappings();
  EXPECT_LT(contents(id).size(), invalid.size());
  remove(id);
  ASSERT_NE(add(0), 0u);
  zones = PADGetDeadZones(0);
  ASSERT_NE(zones, nullptr);
  EXPECT_EQ(zones->stickDeadZone, 5432);
}

TEST_F(PadMappingTest, MissingAdapterSlotDoesNotOverwriteOccupiedSlotAfterMove) {
  const auto first = add(0, true);
  ASSERT_NE(first, 0u);
  bind(0, SDL_GAMEPAD_BUTTON_WEST);
  PADSerializeMappings();
  const auto saved = contents(first);
  ASSERT_FALSE(saved.empty());
  remove(first);
  const auto moved = add(1, true);
  ASSERT_NE(moved, 0u);
  u32 count = 0;
  ASSERT_NE(PADGetButtonMappings(1, &count), nullptr);
  aurora::gamepad::set_player_index(moved, 0);
  PADSerializeMappings();
  EXPECT_EQ(contents(moved), saved);
  expect_binding(0, SDL_GAMEPAD_BUTTON_WEST);
}

TEST_F(PadMappingTest, AdapterWritesPreserveOtherPortSlots) {
  constexpr std::array<SDL_GamepadButton, PAD_CHANMAX> buttons{
      SDL_GAMEPAD_BUTTON_NORTH, SDL_GAMEPAD_BUTTON_EAST, SDL_GAMEPAD_BUTTON_WEST, SDL_GAMEPAD_BUTTON_BACK};
  std::array<SDL_JoystickID, PAD_CHANMAX> ports{};
  for (u32 port = 0; port < PAD_CHANMAX; ++port) {
    ports[port] = add(port, true);
    ASSERT_NE(ports[port], 0u);
    bind(port, buttons[port]);
  }
  PADSerializeMappings();
  const auto saved = contents(ports[0]);
  constexpr size_t headerSize = 32;
  constexpr size_t slotSize = sizeof(PADDeadZones) + sizeof(PADButtonMapping) * PAD_BUTTON_COUNT +
                            sizeof(PADAxisMapping) * PAD_AXIS_COUNT;
  ASSERT_EQ(saved.size(), headerSize + slotSize * PAD_CHANMAX);
  for (u32 port = 1; port < PAD_CHANMAX; ++port) {
    remove(ports[port]);
  }
  ASSERT_NE(add(-1, true), 0u);
  bind(0, SDL_GAMEPAD_BUTTON_START);
  PADSerializeMappings();
  const auto updated = contents(ports[0]);
  ASSERT_EQ(updated.size(), saved.size());
  EXPECT_TRUE(std::equal(saved.begin() + headerSize + slotSize, saved.end(),
                         updated.begin() + headerSize + slotSize));

  for (const auto id : ports) {
    if (aurora::gamepad::g_GameControllers.contains(id)) {
      remove(id);
    }
  }
  for (u32 port = 0; port < PAD_CHANMAX; ++port) {
    ASSERT_NE(add(port, true), 0u);
    expect_binding(port, port == 0 ? SDL_GAMEPAD_BUTTON_START : buttons[port]);
  }
}

} // namespace
