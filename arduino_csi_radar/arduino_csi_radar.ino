/*
 * Sannidhi CSI capture — ESP32 DevKit V1 / Arduino IDE
 *
 * This sketch sends complete 232-feature rows at 25 Hz:
 * CSI_TENSOR,timestamp,frame,<116 amplitudes>,<116 phases>
 *
 * Important: Serial Monitor and Python must both use 921600 baud.
 */

#include <WiFi.h>
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include <math.h>

static const char *WIFI_SSID = "radar";
static const char *WIFI_PASSWORD = "12345678";
static const uint16_t SUBCARRIERS = 116;
static const uint32_t SERIAL_BAUD = 921600;
static const uint32_t OUTPUT_INTERVAL_MS = 40;  // 25 complete samples/second

struct CsiSample {
  int64_t timestamp_us;
  uint32_t frame_number;
  uint16_t pairs;
  int8_t iq[SUBCARRIERS * 2];  // ESP32 CSI order: imag, real
};

static QueueHandle_t csi_queue;
static uint32_t frame_count = 0;

void csi_cb(void *ctx, wifi_csi_info_t *info) {
  if (info == nullptr || info->len == 0 || csi_queue == nullptr) return;

  const int offset = info->first_word_invalid ? 4 : 0;
  const int available_bytes = info->len - offset;
  if (available_bytes < 2) return;

  CsiSample sample = {};
  sample.timestamp_us = esp_timer_get_time();
  sample.frame_number = ++frame_count;
  sample.pairs = min((uint16_t)(available_bytes / 2), SUBCARRIERS);
  memcpy(sample.iq, (const int8_t *)info->buf + offset, sample.pairs * 2);

  // The CSI callback runs in the Wi-Fi task.  Never print/format a long row
  // here: keep only the newest sample and let loop() serialize it safely.
  xQueueOverwrite(csi_queue, &sample);
}

bool append_number(char *line, size_t capacity, size_t *used, float value) {
  int written = snprintf(line + *used, capacity - *used, ",%.2f", value);
  if (written < 0 || (size_t)written >= capacity - *used) return false;
  *used += (size_t)written;
  return true;
}

void send_sample(const CsiSample &sample) {
  char line[2200];
  size_t used = (size_t)snprintf(line, sizeof(line), "CSI_TENSOR,%lld,%u",
                                 sample.timestamp_us, sample.frame_number);
  if (used >= sizeof(line)) return;

  for (uint16_t i = 0; i < SUBCARRIERS; ++i) {
    float amplitude = 0.0f;
    if (i < sample.pairs) {
      float imag = (float)sample.iq[i * 2];
      float real = (float)sample.iq[i * 2 + 1];
      amplitude = sqrtf(real * real + imag * imag);
    }
    if (!append_number(line, sizeof(line), &used, amplitude)) return;
  }

  for (uint16_t i = 0; i < SUBCARRIERS; ++i) {
    float phase = 0.0f;
    if (i < sample.pairs) {
      float imag = (float)sample.iq[i * 2];
      float real = (float)sample.iq[i * 2 + 1];
      phase = atan2f(imag, real);
    }
    if (!append_number(line, sizeof(line), &used, phase)) return;
  }

  if (used + 1 >= sizeof(line)) return;
  line[used++] = '\n';
  Serial.write((const uint8_t *)line, used);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1000);
  Serial.println("Sannidhi CSI capture starting");

  csi_queue = xQueueCreate(1, sizeof(CsiSample));
  if (csi_queue == nullptr) {
    Serial.println("ERROR: CSI queue could not be created");
    return;
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  const uint32_t deadline = millis() + 15000;
  while (WiFi.status() != WL_CONNECTED && millis() < deadline) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("ERROR: Wi-Fi connection failed; check SSID/password/router");
    return;
  }
  Serial.printf("Connected on channel %d\n", WiFi.channel());

  wifi_csi_config_t config = {};
  config.lltf_en = true;
  config.htltf_en = true;
  config.ltf_merge_en = true;
  config.channel_filter_en = true;

  if (esp_wifi_set_csi_config(&config) != ESP_OK ||
      esp_wifi_set_csi_rx_cb(csi_cb, nullptr) != ESP_OK ||
      esp_wifi_set_csi(true) != ESP_OK) {
    Serial.println("ERROR: CSI setup failed");
    return;
  }
  Serial.println("CSI ready: 116 amplitudes + 116 phases at 25 Hz");
}

void loop() {
  static uint32_t last_sent_ms = 0;
  CsiSample sample;

  if (millis() - last_sent_ms >= OUTPUT_INTERVAL_MS &&
      xQueueReceive(csi_queue, &sample, 0) == pdPASS) {
    last_sent_ms = millis();
    send_sample(sample);
  }
  delay(1);
}
