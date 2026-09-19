#include "driver/uart.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static const char *TAG = "Sannidhi_Tensor";
static uint32_t frame_count = 0;
#define CSI_SUBCARRIERS 116

void csi_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || info->len == 0)
    return;

  frame_count++;
  int offset = info->first_word_invalid ? 4 : 0;
  int8_t *wave_data = (int8_t *)info->buf + offset;
  int num_subcarriers = (info->len - offset) / 2;
  int64_t ts = esp_timer_get_time();

  printf("CSI_TENSOR,%lld,%u", ts, frame_count);

  /*
   * The ML contract is exactly [116 amplitudes, 116 phases].  Do not use a
   * single feature counter here: on packets with >116 subcarriers that used
   * to emit 128 amplitudes followed by only 104 phases, so Python split the
   * middle of the amplitude vector as phase.  That silently corrupts every
   * inference window.
   */
  for (int i = 0; i < CSI_SUBCARRIERS; i++) {
    if (i >= num_subcarriers) {
      printf(",0.00");
      continue;
    }
    float real = (float)wave_data[i * 2 + 1];
    float imag = (float)wave_data[i * 2];
    printf(",%.2f", sqrtf(real * real + imag * imag));
  }

  for (int i = 0; i < CSI_SUBCARRIERS; i++) {
    if (i >= num_subcarriers) {
      printf(",0.00");
      continue;
    }
    float real = (float)wave_data[i * 2 + 1];
    float imag = (float)wave_data[i * 2];
    printf(",%.2f", atan2f(imag, real));
  }

  printf("\n");
}

void start_radar() {
  wifi_csi_config_t csi_config = {.lltf_en = true,
                                  .htltf_en = true,
                                  .ltf_merge_en = true,
                                  .channel_filter_en = true};

  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(csi_cb, NULL);
  esp_wifi_set_csi(true);

  ESP_LOGI(TAG, "Radar Online: Outputting 232-feature Tensors.");
}

static void event_handler(void *arg, esp_event_base_t base, int32_t id,
                          void *data) {
  if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
    start_radar();
  }
}

void app_main(void) {
  uart_set_baudrate(UART_NUM_0, 921600); // ✅ correct place

  nvs_flash_init();
  esp_netif_init();
  esp_event_loop_create_default();
  esp_netif_create_default_wifi_sta();

  esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                      &event_handler, NULL, NULL);

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  esp_wifi_init(&cfg);
  esp_wifi_set_mode(WIFI_MODE_STA);

  wifi_config_t wifi_config = {
      .sta = {.ssid = "radar", .password = "12345678"}};

  esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
  esp_wifi_start();
  esp_wifi_set_promiscuous(true);
}
