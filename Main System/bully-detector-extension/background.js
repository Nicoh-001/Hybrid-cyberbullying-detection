const API_URLS = [
    'http://localhost:5000/predict',
];

let currentApiUrlIndex = 0;
const detectionCache = new Map();

async function testApiEndpoint(url) {
    try {
        const healthUrl = url.replace('/predict', '/health');
        const response = await fetch(healthUrl, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            cache: 'no-store',
            signal: AbortSignal.timeout(3000)
        });
        
        if (!response.ok) return false;
        
        const data = await response.json();
        return data.status === 'healthy';
    } catch (error) {
        console.debug(`API health check failed for ${url}:`, error.message);
        return false;
    }
}

let lastWorkingUrl = null;
async function getWorkingApiUrl() {
    if (lastWorkingUrl && await testApiEndpoint(lastWorkingUrl)) {
        return lastWorkingUrl;
    }

    for (let i = 0; i < API_URLS.length; i++) {
        if (await testApiEndpoint(API_URLS[i])) {
            currentApiUrlIndex = i;
            lastWorkingUrl = API_URLS[i];
            return API_URLS[i];
        }
    }
    
    throw new Error('No available API endpoints');
}

async function detectCyberbullying(text) {
    try {
        const apiUrl = await getWorkingApiUrl();
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
            cache: 'no-store',
            signal: AbortSignal.timeout(8000) 
        });

        if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
        }

        const data = await response.json();
        
        if (!data || typeof data.prediction === 'undefined') {
            throw new Error('Invalid API response format');
        }

        if (!data.explanations) {
            data.explanations = [
                {
                    feature: 'unknown',
                    description: 'No explanation available',
                    impact: 0,
                    direction: 'neutral'
                }
            ];
        }

        return data;
    } catch (error) {
        console.error('Detection failed:', error);
        return {
            error: error.message,
            status: 'error',
            prediction: 0, 
            confidence: 0,
            explanations: [
                {
                    feature: 'error',
                    description: error.message,
                    impact: 0,
                    direction: 'neutral'
                }
            ]
        };
    }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const handleRequest = async () => {
        try {
            if (request.action === 'detectCyberbullying') {
                const result = await detectCyberbullying(request.text);
                return result;
            }
            else if (request.action === 'analyzeText') {
                const result = await detectCyberbullying(request.payload);
                return result;
            }
            else if (request.action === 'getSettings') {
                const result = await chrome.storage.local.get(['enabled', 'threshold', 'cleanColor', 'bullyColor', 'severeColor', 'privacyMode']);
                return {
                    enabled: result.enabled !== false,
                    threshold: result.threshold || 0.7,
                    cleanColor: result.cleanColor || '#28a745',
                    bullyColor: result.bullyColor || '#ffc107',
                    severeColor: result.severeColor || '#dc3545',
                    privacyMode: result.privacyMode !== false,
                    apiStatus: lastWorkingUrl ? 'connected' : 'disconnected',
                    apiUrl: lastWorkingUrl || 'none'
                };
            }
            else if (request.action === 'saveSettings') {
                await chrome.storage.local.set({
                    enabled: request.enabled,
                    threshold: request.threshold,
                    cleanColor: request.cleanColor,
                    bullyColor: request.bullyColor,
                    severeColor: request.severeColor,
                    privacyMode: request.privacyMode
                });
                return { status: 'success' };
            }
            else if (request.action === 'checkApiStatus') {
                try {
                    await getWorkingApiUrl();
                    return {
                        status: 'connected',
                        apiUrl: lastWorkingUrl
                    };
                } catch (error) {
                    return {
                        status: 'disconnected',
                        error: error.message
                    };
                }
            }
            else if (request.action === 'updateCounts') {
                chrome.tabs.query({}, tabs => {
                    tabs.forEach(tab => {
                        if (tab.id) {
                            chrome.tabs.sendMessage(tab.id, {
                                action: 'updateCountDisplay',
                                counts: request.counts
                            }).catch(() => {});
                        }
                    });
                });
                return { status: 'success' };
            }
            else if (request.action === 'logFeedback') {
                console.log('User feedback:', request);
                return { status: 'success' };
            }
            else if (request.action === 'setBadge') {
                chrome.action.setBadgeText({ text: request.text });
                chrome.action.setBadgeBackgroundColor({ color: request.color });
                return { status: 'success' };
            }
            else if (request.action === 'resetCounts') {
                chrome.tabs.query({}, tabs => {
                    tabs.forEach(tab => {
                        if (tab.id) {
                            chrome.tabs.sendMessage(tab.id, {
                                action: 'clearCounts'
                            }).catch(() => {});
                        }
                    });
                });
                return { status: 'success' };
            }
        } catch (error) {
            console.error('Background error:', error);
            return {
                error: error.message,
                status: 'error'
            };
        }
    };

    handleRequest().then(sendResponse);
    return true; 
});

async function initialize() {
    let attempts = 0;
    const maxAttempts = 3;
    
    while (attempts < maxAttempts) {
        try {
            await getWorkingApiUrl();
            console.log('Using API endpoint:', lastWorkingUrl);
            break;
        } catch (error) {
            attempts++;
            if (attempts >= maxAttempts) {
                console.error('Failed to connect to API after', maxAttempts, 'attempts');
            } else {
                await new Promise(resolve => setTimeout(resolve, 2000 * attempts));
            }
        }
    }

    const { enabled, threshold, cleanColor, bullyColor, severeColor, privacyMode } = await chrome.storage.local.get(['enabled', 'threshold', 'cleanColor', 'bullyColor', 'severeColor', 'privacyMode']);
    if (enabled === undefined) await chrome.storage.local.set({ enabled: true });
    if (threshold === undefined) await chrome.storage.local.set({ threshold: 0.7 });
    if (cleanColor === undefined) await chrome.storage.local.set({ cleanColor: '#28a745' });
    if (bullyColor === undefined) await chrome.storage.local.set({ bullyColor: '#ffc107' });
    if (severeColor === undefined) await chrome.storage.local.set({ severeColor: '#dc3545' });
    if (privacyMode === undefined) await chrome.storage.local.set({ privacyMode: true });

    // Set up count reset timer
    setInterval(() => {
        chrome.runtime.sendMessage({ action: 'resetCounts' }).catch(() => {});
    }, 10 * 60 * 1000); // 10 minutes
}

initialize();