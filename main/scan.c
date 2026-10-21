#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include <math.h>
#include <stdio.h>

// ==============================================================================
// 📡 WI-FI & RADAR CONFIGURATION (Edit Wi-Fi Name, Password & Laptop IP here)
// ==============================================================================
#define WIFI_SSID "NARZO 70 5G"        // Your Wi-Fi / Hotspot Name
#define WIFI_PASS "kvqw1445"           // Your Wi-Fi / Hotspot Password
#define CSI_RECEIVER_IP "10.94.84.204" // Explicit Laptop IP
#define CSI_RECEIVER_PORT 5555

#define CSI_FEATURES 52
#define CSI_QUEUE_LENGTH 32
#define WIFI_READY_BIT BIT0

// 52 active OFDM data subcarrier bins for 802.11n HT20 (ESP-Fi-HAR contract)
// Negative subcarriers (-26 to -1): bins 38 to 63 (26 active bins)
// Positive subcarriers (+1 to +26): bins 1 to 26 (26 active bins)
// Total = 52 subcarriers (skipping DC null bin 0 and guard bands 27..37)
static const uint8_t csi_subcarrier_bins[CSI_FEATURES] = {
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 26};

static const char *TAG = "Sannidhi_Tensor";
static uint32_t frame_count = 0;
static QueueHandle_t csi_queue;
static EventGroupHandle_t wifi_events;

typedef struct {
  int64_t timestamp;
  uint32_t frame_number;
  float features[CSI_FEATURES];
} csi_frame_t;

static void csi_callback(void *ctx, wifi_csi_info_t *info) {
  if (!info || info->len < 128 || csi_queue == NULL)
    return;

  frame_count++;
  int8_t *wave_data = (int8_t *)info->buf;
  int num_subcarriers = info->len / 2;

  csi_frame_t frame = {.timestamp = esp_timer_get_time(),
                       .frame_number = frame_count};

  // ESP-Fi-HAR contract: amplitude only, 52 active subcarriers per frame.
  for (int i = 0; i < CSI_FEATURES; i++) {
    int source_bin = csi_subcarrier_bins[i];
    // Compensate for ESP32 hardware limitation where bytes 0-3 (bins 0 & 1) are invalid
    if (source_bin == 1 && info->first_word_invalid) {
      source_bin = 2; // interpolate from adjacent active subcarrier (+2)
    }
    float real = 0.0f;
    float imag = 0.0f;
    if (source_bin < num_subcarriers) {
      real = (float)wave_data[source_bin * 2 + 1];
      imag = (float)wave_data[source_bin * 2];
    }
    frame.features[i] = sqrtf(real * real + imag * imag);
  }

  // Do not perform UART or socket I/O inside the Wi-Fi callback.
  xQueueSend(csi_queue, &frame, 0);
}

static void csi_udp_writer_task(void *arg) {
  xEventGroupWaitBits(wifi_events, WIFI_READY_BIT, pdFALSE, pdTRUE,
                      portMAX_DELAY);

  csi_frame_t frame;
  char packet[1800];
  int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

  if (sock < 0) {
    ESP_LOGE(TAG, "Could not create UDP socket");
    vTaskDelete(NULL);
    return;
  }

  // Enable broadcast so CSI packets reach all laptops/devices on the Wi-Fi
  int broadcast_enable = 1;
  setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast_enable,
             sizeof(broadcast_enable));

  struct sockaddr_in receiver = {0};
  receiver.sin_family = AF_INET;
  receiver.sin_port = htons(CSI_RECEIVER_PORT);
  receiver.sin_addr.s_addr = inet_addr(CSI_RECEIVER_IP);

  ESP_LOGI(TAG, "UDP target %s:%d (Broadcast Enabled)", CSI_RECEIVER_IP,
           CSI_RECEIVER_PORT);

  while (true) {
    if (xQueueReceive(csi_queue, &frame, portMAX_DELAY) != pdTRUE)
      continue;

    int used = snprintf(packet, sizeof(packet), "CSI_TENSOR,%lld,%lu",
                        frame.timestamp, (unsigned long)frame.frame_number);

    for (int i = 0; i < CSI_FEATURES && used < (int)sizeof(packet) - 16; i++) {
      used += snprintf(packet + used, sizeof(packet) - used, ",%.2f",
                       frame.features[i]);
    }

    if (used > 0 && used < (int)sizeof(packet) - 2) {
      packet[used++] = '\n';
      sendto(sock, packet, used, 0, (struct sockaddr *)&receiver,
             sizeof(receiver));
    }
  }
}

static void start_radar(void) {
  wifi_csi_config_t config = {.lltf_en = true,
                              .htltf_en = true,
                              .ltf_merge_en = true,
                              .channel_filter_en = true};

  esp_wifi_set_csi_config(&config);
  esp_wifi_set_csi_rx_cb(csi_callback, NULL);
  esp_wifi_set_csi(true);
  esp_wifi_set_promiscuous(true);
  wifi_promiscuous_filter_t filter = {.filter_mask = WIFI_PROMIS_FILTER_MASK_DATA};
  esp_wifi_set_promiscuous_filter(&filter);
  esp_wifi_set_ps(WIFI_PS_NONE);

  xEventGroupSetBits(wifi_events, WIFI_READY_BIT);
  ESP_LOGI(TAG, "Radar online; CSI is being sent over UDP");
}

static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id,
                               void *data) {
  if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
    ESP_LOGI(TAG, "Connecting to Wi-Fi SSID '%s'...", WIFI_SSID);
    esp_wifi_connect();
  } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
    wifi_event_sta_disconnected_t *disc = (wifi_event_sta_disconnected_t *)data;
    ESP_LOGW(TAG, "Wi-Fi disconnected! Reason: %d. Reconnecting...", disc ? disc->reason : -1);
    esp_wifi_connect();
  } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)data;
    ESP_LOGI(TAG, "Connected! ESP32 IP: " IPSTR, IP2STR(&event->ip_info.ip));
    start_radar();
  }
}

void app_main(void) {
  wifi_events = xEventGroupCreate();
  csi_queue = xQueueCreate(CSI_QUEUE_LENGTH, sizeof(csi_frame_t));
  if (!csi_queue) {
    ESP_LOGE(TAG, "Could not allocate CSI queue");
    return;
  }

  xTaskCreate(csi_udp_writer_task, "csi_udp_writer", 8192, NULL, 5, NULL);

  nvs_flash_init();
  esp_netif_init();
  esp_event_loop_create_default();
  esp_netif_create_default_wifi_sta();

  esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                      &wifi_event_handler, NULL, NULL);
  esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                      &wifi_event_handler, NULL, NULL);

  wifi_init_config_t init_config = WIFI_INIT_CONFIG_DEFAULT();
  esp_wifi_init(&init_config);
  esp_wifi_set_mode(WIFI_MODE_STA);

  wifi_config_t wifi_config = {
      .sta = {
          .ssid = WIFI_SSID,
          .password = WIFI_PASS,
          .threshold.authmode = WIFI_AUTH_WPA2_PSK,
          .pmf_cfg = {
              .capable = true,
              .required = false,
          },
      },
  };

  esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
  esp_wifi_start();
}
