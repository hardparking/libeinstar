// Device behaviour against the simulated scanner (no hardware).
#include <atomic>
#include <chrono>
#include <thread>

#include "check.hpp"
#include "einstar/device.hpp"
#include "fake_scanner.hpp"

using namespace einstar;
using einstar::test::FakeScanner;

namespace {

struct Opened {
  std::unique_ptr<Device> device;
  FakeScanner* scanner = nullptr;  // owned by device
};

Opened open_fake(OpenOptions options = {}, void (*configure)(FakeScanner&) = nullptr) {
  auto fake = std::make_unique<FakeScanner>();
  if (configure) configure(*fake);
  FakeScanner* raw = fake.get();
  options.timeout_ms = 50;
  auto dev = Device::open(std::move(fake), options);
  CHECK(dev.ok());
  return {dev ? std::move(dev.value()) : nullptr, raw};
}

std::vector<std::uint8_t> be32(std::uint32_t v) {
  std::vector<std::uint8_t> b(4);
  write_be32(b, 0, v);
  return b;
}

}  // namespace

TEST(device_handshake_reads_serial) {
  auto o = open_fake();
  if (!o.device) return;
  CHECK(o.device->serial() == "123456789ABCDEF0");
}

TEST(device_getters_decode_replies) {
  auto o = open_fake();
  if (!o.device) return;
  CHECK(o.device->exposure_us(0).value() == 5000);
  CHECK(o.device->gain(2).value() == 200);
  CHECK(o.device->sensor_count().value() == 3);
  CHECK(o.device->temperature_raw().value() == 50);
  CHECK(o.device->firmware_version().value() == "EinScan10_01_SC130_FX3_V2.14_FPGA_V3.7_EN");
  CHECK(o.device->trigger_period().value() == 90000);
  // Selectors are camera masks and requests use the vendor transfer lengths.
  const auto rx = o.scanner->received();
  // rx[0] is the serial query made by Device::open.
  CHECK(rx[0].group == 0x00 && rx[0].opcode == 0x04 && rx[0].wire_size == 64);
  CHECK(rx[1].opcode == 0x22 && rx[1].payload == std::vector<std::uint8_t>{1} && rx[1].wire_size == 16);
  CHECK(rx[2].opcode == 0x26 && rx[2].payload == std::vector<std::uint8_t>{4});
  CHECK(rx[5].group == 0x00 && rx[5].opcode == 0x05 && rx[5].wire_size == 64);
  CHECK(rx[6].opcode == 0x48 && rx[6].wire_size == 50);
}

TEST(device_writes_need_allow_writes) {
  auto ro = open_fake();
  if (!ro.device) return;
  auto s = ro.device->set_gain(0, 300);
  CHECK(!s.ok() && s.error().code == ErrorCode::WritesDisabled);
  CHECK(ro.scanner->received().size() == 1);  // only the serial query

  OpenOptions rw;
  rw.allow_writes = true;
  auto o = open_fake(rw);
  if (!o.device) return;
  CHECK(o.device->set_gain(1, 300).ok());
  CHECK(o.device->gain(1).value() == 300);
  const auto rx = o.scanner->received();
  CHECK(rx[1].opcode == 0x27 && rx[1].payload == (std::vector<std::uint8_t>{2, 0x01, 0x2C}));
}

TEST(device_never_sends_flash_writes_or_reboot) {
  OpenOptions rw;
  rw.allow_writes = true;
  auto o = open_fake(rw);
  if (!o.device) return;
  CHECK(o.device->transact(Opcode::PageWrite).error().code == ErrorCode::WritesDisabled);
  CHECK(o.device->transact(command(SystemOpcode::Reboot)).error().code == ErrorCode::WritesDisabled);
  CHECK(o.scanner->received().size() == 1);  // only the serial query
}

TEST(device_retries_bad_replies_up_to_five_times) {
  auto o = open_fake();
  if (!o.device) return;
  o.scanner->corrupt_replies = 2;
  CHECK(o.device->sensor_count().value() == 3);
  CHECK(o.scanner->received().size() == 1 + 3);
  CHECK(o.scanner->clear_halts() >= 2 * 2 + 2);  // two per failed attempt plus the handshake's

  auto bad = open_fake();
  if (!bad.device) return;
  bad.scanner->corrupt_replies = 100;
  auto r = bad.device->sensor_count();
  CHECK(!r.ok() && r.error().code == ErrorCode::BadReply);
  CHECK(bad.scanner->received().size() == 1 + 5);
}

TEST(device_mask_option_matches_a_masking_scanner) {
  OpenOptions masked;
  masked.mask_commands = true;
  auto o = open_fake(masked, [](FakeScanner& f) { f.require_mask = true; });
  if (!o.device) return;
  CHECK(o.device->exposure_us(1).value() == 5000);

  // Without the mask, opening already fails at the serial query.
  auto fake = std::make_unique<FakeScanner>();
  fake->require_mask = true;
  OpenOptions plain;
  plain.timeout_ms = 50;
  CHECK(!Device::open(std::move(fake), plain).ok());
}

TEST(device_start_and_stop_scanning_follow_vendor_sequence) {
  OpenOptions rw;
  rw.allow_writes = true;
  auto o = open_fake(rw);
  if (!o.device) return;
  CHECK(o.device->start_scanning().ok());
  CHECK(o.device->trigger_counts().value() == std::make_pair(1u, 1u));
  CHECK(o.device->stop_scanning().ok());
  auto rx = o.scanner->received();
  rx.erase(rx.begin());  // serial query
  CHECK(rx.size() == 6);
  CHECK(rx[0].opcode == 0x49 && rx[0].payload == be32(200000) && rx[0].wire_size == 50);
  CHECK(rx[1].opcode == 0x41 && rx[1].payload == std::vector<std::uint8_t>{0x11} && rx[1].wire_size == 10);
  CHECK(rx[2].opcode == 0x49 && rx[2].payload == be32(90000));
  CHECK(rx[3].opcode == 0x40);
  CHECK(rx[4].opcode == 0x49 && rx[4].payload == be32(200000));
  CHECK(rx[5].opcode == 0x41 && rx[5].payload == std::vector<std::uint8_t>{0x00});
}

TEST(device_reads_custom_info_across_pages) {
  auto o = open_fake({}, [](FakeScanner& f) {
    for (int p = 0; p < 2; ++p) {
      std::vector<std::uint8_t> page(kPageSize);
      for (std::size_t i = 0; i < page.size(); ++i) page[i] = static_cast<std::uint8_t>(p * 7 + i);
      f.pages.push_back(page);
    }
  });
  if (!o.device) return;
  auto blob = o.device->read_custom_info(kDeviceInfoSize);
  CHECK(blob.ok());
  if (!blob) return;
  CHECK(blob.value().size() == kDeviceInfoSize);
  CHECK(blob.value()[10] == 10);
  CHECK(blob.value()[kPageSize + 10] == static_cast<std::uint8_t>(7 + 10));
  const auto rx = o.scanner->received();
  CHECK(rx.size() == 3 && rx[1].opcode == 0x57 && rx[2].payload == (std::vector<std::uint8_t>{0, 1}));
  // Missing pages surface as a device status error, not garbage data.
  CHECK(!o.device->read_page(9).ok());
}

TEST(device_streams_reassembled_frames) {
  auto o = open_fake();
  if (!o.device) return;
  std::atomic<int> frames{0};
  std::atomic<unsigned> camera{99};
  std::atomic<std::uint32_t> frame_id{0};
  CHECK(o.device->start_stream(
                    [&](Frame&& f) {
                      camera = f.info.camera_index().value_or(98);
                      frame_id = f.info.frame_id;
                      ++frames;
                    },
                    64)
            .ok());
  for (int part = 0; part < 2; ++part) {
    std::vector<std::uint8_t> t(kStreamHeaderSize + 32, 0);
    t[kOffStreamCamera] = 4 << 3;  // colour camera
    t[kOffStreamLastFlag] = part == 1;
    write_be32(t, kOffStreamFrameId, 77);
    o.scanner->queue_stream_transfer(t);
  }
  for (int i = 0; i < 200 && frames == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  o.device->stop_stream();
  CHECK(frames == 1);
  CHECK(camera == 2u);
  CHECK(frame_id == 77u);
}
