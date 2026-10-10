idf_component_register(
    SRCS "scan.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        esp_wifi
        esp_event
        nvs_flash
        esp_timer
        freertos
        lwip
)