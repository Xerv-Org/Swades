import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

VERSION = "1.0"

PROMPT = (
    "Camera head on one pan servo. hw_look returns a fresh frame; hw_pan turns the head (absolute degrees, or a signed offset with relative=true) and returns the real angle; hw_status reports the device. "
    "Learn which way increasing degrees turns by moving and comparing frames. Moves are clamped and split into steps, so verify with hw_look instead of assuming. "
    "To follow a target: look, locate it horizontally, pan a proportional correction, look again. Describe only what frames show."
)
CHECK = "Camera: take your own fresh frame with hw_look and judge only by what you see."

FIRMWARE = r"""#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
static const char *WIFI_SSID = "@SSID@";
static const char *WIFI_PASS = "@PASS@";
static const char *API_KEY = "@KEY@";
static const int SERVO_PIN = @PIN@;
static const int PULSE_MIN = @PMIN@;
static const int PULSE_MAX = @PMAX@;
static const int ANGLE_MIN = @AMIN@;
static const int ANGLE_MAX = @AMAX@;
WebServer server(80);
Servo pan;
int angle = @AHOME@;
bool authorised() {
  if (server.arg("key") == API_KEY) return true;
  server.send(401, "text/plain", "bad key");
  return false;
}
void handleCapture() {
  if (!authorised()) return;
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { server.send(503, "text/plain", "capture failed"); return; }
  WiFiClient client = server.client();
  client.printf("HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\nConnection: close\r\n\r\n", (unsigned)fb->len);
  client.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}
void handlePan() {
  if (!authorised()) return;
  if (!server.hasArg("deg")) { server.send(400, "text/plain", "deg required"); return; }
  angle = constrain(server.arg("deg").toInt(), ANGLE_MIN, ANGLE_MAX);
  pan.write(angle);
  server.send(200, "application/json", String("{\"deg\":") + angle + "}");
}
void handleStatus() {
  if (!authorised()) return;
  server.send(200, "application/json", String("{\"deg\":") + angle + ",\"rssi\":" + WiFi.RSSI() + ",\"heap\":" + ESP.getFreeHeap() + "}");
}
void setup() {
  Serial.begin(115200);
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_7;
  c.ledc_timer = LEDC_TIMER_3;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM; c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_VGA;
  c.jpeg_quality = 12;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) { Serial.println("camera init failed"); while (true) delay(1000); }
  ESP32PWM::allocateTimer(0);
  pan.setPeriodHertz(50);
  pan.attach(SERVO_PIN, PULSE_MIN, PULSE_MAX);
  pan.write(angle);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println(WiFi.localIP());
  server.on("/capture", handleCapture);
  server.on("/pan", handlePan);
  server.on("/status", handleStatus);
  server.begin();
}
void loop() {
  if (WiFi.status() != WL_CONNECTED) { WiFi.reconnect(); delay(2000); return; }
  server.handleClient();
}
"""


def firmware(get):
    missing = [n for n in ("HW_WIFI_SSID", "HW_WIFI_PASS", "HW_KEY") if not get(n)]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}")

    def quote(text):
        return text.replace("\\", "\\\\").replace('"', '\\"')

    def whole(name, default):
        return str(int(float(get(name) or default)))

    fields = {
        "@SSID@": quote(get("HW_WIFI_SSID")),
        "@PASS@": quote(get("HW_WIFI_PASS")),
        "@KEY@": quote(get("HW_KEY")),
        "@PIN@": whole("HW_PIN", 13),
        "@PMIN@": whole("HW_PULSE_MIN", 500),
        "@PMAX@": whole("HW_PULSE_MAX", 2400),
        "@AMIN@": whole("HW_MIN", 10),
        "@AMAX@": whole("HW_MAX", 170),
        "@AHOME@": whole("HW_HOME", 90),
    }
    text = FIRMWARE
    for token, value in fields.items():
        text = text.replace(token, value)
    return text


def setup(api):
    url = (api.get("HW_URL") or "").rstrip("/")
    capture = api.get("HW_CAPTURE_URL") or ""
    key = api.get("HW_KEY") or ""
    if not url and not capture:
        raise api.AgentError("set HW_URL or HW_CAPTURE_URL")
    lo, hi = api.num("HW_MIN", 10.0), api.num("HW_MAX", 170.0)
    step, gap, settle = api.num("HW_STEP", 45.0), api.num("HW_GAP", 0.3), api.num("HW_SETTLE", 0.4)
    timeout, cap = api.num("HW_TIMEOUT", 10), api.num("HW_IMAGE_BYTES", 1500000)
    head = {"angle": None, "last": 0.0}

    def fetch(target, params, limit):
        if not target.startswith(("http://", "https://")):
            raise api.ToolError("device url must be http(s)")
        if key:
            params = {**params, "key": key}
        if params:
            target += ("&" if "?" in target else "?") + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(urllib.request.Request(target), timeout=timeout) as reply:
                data = reply.read(limit + 1)
        except urllib.error.HTTPError as exc:
            raise api.ToolError(f"device http {exc.code}")
        except OSError as exc:
            raise api.ToolError(f"device unreachable: {exc}")
        if len(data) > limit:
            raise api.ToolError("device reply too large")
        return data

    def number(raw):
        try:
            return float(json.loads(raw)["deg"])
        except (ValueError, KeyError, TypeError):
            raise api.ToolError("bad device reply")

    def status(_=None):
        if not url:
            raise api.ToolError("HW_URL not set; camera only")
        raw = fetch(url + "/status", {}, 4096)
        head["angle"] = number(raw)
        data = json.loads(raw)
        return f"angle={head['angle']:g} limits={lo:g}-{hi:g} rssi={data.get('rssi', '?')} heap={data.get('heap', '?')}"

    def look(_):
        raw = fetch(capture or url + "/capture", {}, cap)
        media = api.image_type(raw)
        if not media:
            raise api.ToolError("device did not return an image")
        where = "unknown" if head["angle"] is None else f"{head['angle']:g}"
        return api.Img(media, base64.b64encode(raw).decode("ascii"), f"frame, head at {where} deg")

    def pan(a):
        if head["angle"] is None:
            status()
        want = head["angle"] + float(a["deg"]) if a.get("relative") else float(a["deg"])
        goal = max(lo, min(hi, want))
        move = max(-step, min(step, goal - head["angle"]))
        wait = gap - (time.monotonic() - head["last"])
        if wait > 0:
            time.sleep(wait)
        head["angle"] = number(fetch(url + "/pan", {"deg": int(round(head["angle"] + move))}, 4096))
        head["last"] = time.monotonic()
        if settle > 0:
            time.sleep(settle)
        notes = []
        if goal != want:
            notes.append(f"clamped to {goal:g}")
        if abs(goal - head["angle"]) > 0.5:
            notes.append(f"step limit {step:g}; call again to continue toward {goal:g}")
        return f"angle={head['angle']:g}" + ("; " + "; ".join(notes) if notes else "")

    def brief():
        if not url:
            return "camera only, no servo"
        try:
            return status()
        except api.ToolError as exc:
            return f"device unreachable: {exc}"

    api.tool("hw_look", "Fresh camera frame as an image.", {}, look, safe=True)
    if url:
        api.tool("hw_pan", "Turn the head. deg is absolute degrees, or a signed offset when relative=true. Returns the real resulting angle.", {"deg": api.need("number"), "relative": api.opt("boolean")}, pan)
        api.tool("hw_status", "Device angle, limits, Wi-Fi signal, memory.", {}, status, safe=True)
    api.prompt(PROMPT)
    api.check(CHECK)
    api.state(brief)


if __name__ == "__main__":
    print(firmware(os.environ.get))
