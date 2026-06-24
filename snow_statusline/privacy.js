// 将文件复制到 ~/.snow/plugin/statusline/

const REQUEST_TIMEOUT_MS = 1000;

function createPrivacyItem(isOn) {
  const text = isOn ? " Privacy" : " Privacy";

  return {
    id: "builtin.privacy",
    text,
    detailedText: `Privacy service: ${text}`,
    color: isOn ? "green" : "red",
  };
}

function shouldShowPrivacyStatus(context) {
  const privacy = context?.system?.privacy;

  return Boolean(privacy?.enabled && privacy.mode === "api");
}

function getHealthUrl(context) {
  const apiUrl = context?.system?.privacy?.apiUrl;

  if (typeof apiUrl !== "string" || apiUrl.trim() === "") {
    return undefined;
  }

  return apiUrl.trim().replace(/\/mask$/, "/health");
}

async function isPrivacyOn(healthUrl) {
  if (!healthUrl) {
    return false;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(healthUrl, {
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
  id: "builtin.privacy",
  refreshIntervalMs: 10_000,
  async getItems(context) {
    if (!shouldShowPrivacyStatus(context)) {
      return undefined;
    }

    const isOn = await isPrivacyOn(getHealthUrl(context));
    return createPrivacyItem(isOn);
  },
};
