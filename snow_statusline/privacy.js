// 将文件复制到 ~/.snow/plugin/statusline/

const HEALTH_URL = "http://127.0.0.1:4175/health";  // 隐私脱敏服务地址
const REQUEST_TIMEOUT_MS = 1000;

function createPrivacyItem(isOn) {
  const text = isOn ? " Privacy" : " Privacy";

  return {
    id: "custom-privacy-status",
    text,
    detailedText: `Privacy service: ${text}`,
    color: isOn ? "green" : "red",
  };
}

async function isPrivacyOn() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(HEALTH_URL, {
      method: "GET",
      signal: controller.signal,
    });

    return response.status === 200;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

export default {
  id: "custom.privacy-status",
  refreshIntervalMs: 10_000,
  async getItems() {
    const isOn = await isPrivacyOn();
    return createPrivacyItem(isOn);
  },
};
