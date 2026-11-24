const HIGHLIGHT_CLASS = 'cyberbullying-highlight';
const HIGHLIGHT_BULLY_CLASS = 'cyberbullying-bully';
const HIGHLIGHT_SEVERE_CLASS = 'cyberbullying-severe';
const HIGHLIGHT_CLEAN_CLASS = 'cyberbullying-clean';
const DETECTION_HISTORY_SIZE = 100;

const PLATFORM_CONFIG = {
    'twitter.com': {
        textContainers: [
            'article[data-testid="tweet"] div[data-testid="tweetText"]',
            'div[data-testid="tweet"] div[lang]'
        ],
        ignoreSelectors: [
            '[aria-*]',
            '[role="navigation"]',
            '[role="button"]',
            '[href*="/i/web/"]',
            'time',
            'svg',
            '[data-testid*="icon"]',
            'header',
            'nav',
            'aside',
            '[data-testid="User-Name"]'
        ],
        hideSelectors: [
            'img[src*="profile_images"]',
            '[data-testid="User-Name"]',
            '[data-testid="UserAvatar-Container"]',
            '[data-testid="tweet"] [role="link"]', // Additional username selector
            '[data-testid="tweet"] [dir="ltr"] span', // Additional username selector
            'video',
            'img[src*="media"]',
            '[data-testid="tweetPhoto"]',
            '[aria-label="Image"]',
            '[aria-label="Video"]'
        ],
        'web.telegram.org': {
            textContainers: [
                '.message:not(.service) .text'
            ],
            ignoreSelectors: [
                '[class*="reply"]',
                '[class*="forward"]',
                '[class*="time"]'
            ],
            hideSelectors: [
                'img[src*="userpic"]',
                '.userpic',
                '.peer-title',
                '.message .video',
                '.message .photo'
            ],
            ignorePatterns: [
                /^\d{1,2}:\d{2}$/,
                /^Saved Messages$/
            ],
            minTextLength: 10,
            maxTextLength: 1000,
            minUniqueWords: 3
        },
        ignorePatterns: [
            /^@\w+$/,
            /^#\w+$/,
            /^[\d,]+ (likes|retweets|replies)$/i,
            /^View \d+ (replies|likes)$/i,
            /^Show this thread$/i,
            /^Keyboard shortcuts$/i,
            /^Notifications?$/i,
            /^Communities?$/i,
            /^To view keyboard shortcuts/i
        ],
        minTextLength: 20,
        maxTextLength: 280,
        minUniqueWords: 3
    },
    'facebook.com': {
        textContainers: [
            'div[data-ad-preview="message"]',
            'div[dir="auto"][role="article"]'
        ],
        ignoreSelectors: [
            '[role="navigation"]',
            '[aria-label="Comment"]',
            'a[role="link"][tabindex="0"]',
            '[aria-label="Like"]',
            '[aria-label="Send"]'
        ],
        hideSelectors: [
            'img[alt*="profile picture"]',
            'a[role="link"][tabindex="0"] span',
            'div[data-visualcompletion="ignore"]',
            'span[dir="auto"]',
            'h2 strong span',
            '[aria-label*="profile picture"]',
            '[aria-label*="user"]',
            '[aria-label*="account"]',
            '[role="article"] [dir="auto"]',
            '[data-pagelet="ProfileTiles"]',
            '[data-pagelet="ProfilePhotos"]',
            'video',
            'img[src*="fbcdn.net"]',
            '[data-visualcompletion="media-video"]',
            '[data-visualcompletion="media-image"]',
            '[aria-label="Photo"]',
            '[aria-label="Video"]',
            '[data-visualcompletion="video"]'
        ],
        ignorePatterns: [
            /^@\w+$/,
            /^Like$/i,
            /^Comment$/i,
            /^Share$/i
        ],
        minTextLength: 15,
        maxTextLength: 1000,
        minUniqueWords: 3
    },
    'instagram.com': {
        textContainers: [
            'div[role="button"] + div span',
            'div._a9zs > span'
        ],
        ignoreSelectors: [
            'header',
            'nav',
            '[role="navigation"]',
            'a[role="link"]',
            'svg'
        ],
        hideSelectors: [
            'img[data-testid="user-avatar"]',
            'a[role="link"][tabindex="0"]',
            'header img',
            'header h2',
            '[class*="author"]',
            '[class*="username"]',
            'video',
            'img[src*="cdninstagram.com"]',
            'div[role="button"][aria-label*="photo"]',
            'div[role="button"][aria-label*="video"]',
            '[aria-label="Photo"]',
            '[aria-label="Video"]'
        ],
        ignorePatterns: [
            /^@\w+$/,
            /^Like$/i,
            /^\d+ likes$/i,
            /^View all \d+ comments$/i
        ],
        minTextLength: 15,
        maxTextLength: 500,
        minUniqueWords: 3
    },
    'default': {
        textContainers: ['p', 'div', 'span'],
        ignoreSelectors: ['a', 'button', 'nav', 'header'],
        hideSelectors: [
            'img[src*="avatar"]',
            'img[alt*="profile"]',
            '[class*="user"]',
            '[class*="username"]',
            'video',
            'img[src*="http"]',
            '[class*="media"]',
            '[class*="image"]',
            '[class*="video"]',
            '[aria-label*="image"]',
            '[aria-label*="video"]'
        ],
        ignorePatterns: [
            /^@\w+$/,
            /^#\w+$/,
            /^[\d,]+ (likes|views|comments)$/i
        ],
        minTextLength: 10,
        maxTextLength: 500,
        minUniqueWords: 3
    }
};

const SCAN_CHUNK_SIZE = 3;
const SCAN_DELAY_MS = 300;

let isScanning = false;
let scrollHandler = null;
let mutationObserver = null;
let currentCounts = { clean: 0, bully: 0, severe: 0 };
let detectedTexts = { clean: [], bully: [], severe: [] };
const analyzedElements = new WeakSet();
let detectionHistory = new Set();
let currentDetection = null;

function initContentScript() {
    if (window.hasCyberbullyingContentScript) return;
    window.hasCyberbullyingContentScript = true;

    // Immediately check privacy mode and apply if needed
    chrome.runtime.sendMessage({ action: 'getSettings' }, (response) => {
        if (response?.privacyMode !== false) {
            document.body.classList.add('cyberbullying-privacy-mode');
            hideSensitiveElements(PLATFORM_CONFIG[window.location.hostname] || PLATFORM_CONFIG.default);
        }
    });

    // Initialize analysis with delay
    setTimeout(() => safeAnalyzePage(), 1500);
    setupScrollAnalysis();
    setupMutationObserver();
    
    chrome.runtime.onMessage.addListener(handleSettingsUpdate);
}

function handleSettingsUpdate(request, sender, sendResponse) {
    const hostname = window.location.hostname;
    const config = PLATFORM_CONFIG[hostname] || PLATFORM_CONFIG.default;

    if (request.action === 'updateSettings') {
        // Toggle privacy mode
        if (request.privacyMode !== false) {
            document.body.classList.add('cyberbullying-privacy-mode');
            hideSensitiveElements(config);
        } else {
            document.body.classList.remove('cyberbullying-privacy-mode');
            restoreSensitiveElements(config);
        }

        clearHighlights();
        analyzedElements.clear();
        safeAnalyzePage();
    } else if (request.action === 'updateHighlightColors') {
        updateHighlightStyles(request.colors);
    } else if (request.action === 'updateCountDisplay') {
        updateCountBadge(request.counts);
    } else if (request.action === 'togglePrivacyMode') {
        if (request.enabled) {
            document.body.classList.add('cyberbullying-privacy-mode');
            hideSensitiveElements(config);
        } else {
            document.body.classList.remove('cyberbullying-privacy-mode');
            restoreSensitiveElements(config);
        }
    }
}

// Update the hideSensitiveElements function in content.js
function hideSensitiveElements(config) {
    // Apply platform-specific hiding
    if (config.hideSelectors) {
        config.hideSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                el.classList.add('cyberbullying-force-dark');
            });
        });
    }
    
    // Nuclear option for all potential username elements
    const usernameSelectors = [
        '[class*="user"]',
        '[class*="name"]',
        '[class*="handle"]',
        '[data-testid*="User"]',
        '[data-testid*="user"]',
        '[class*="author"]',
        '[class*="screen-name"]',
        '[class*="account"]',
        '[class*="profile"]'
    ];
    
    usernameSelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => {
            if (el.textContent.match(/^@\w+$/) || 
                el.textContent.match(/^[A-Z][a-z]+ [A-Z][a-z]+$/) ||
                el.textContent.match(/^[A-Z][a-z]+$/)) {
                el.classList.add('cyberbullying-force-dark');
            }
        });
    });
    
    // Platform-specific username hiding
    if (window.location.hostname === 'twitter.com') {
        document.querySelectorAll('[data-testid="User-Name"]').forEach(el => {
            el.classList.add('cyberbullying-force-dark');
        });
    }
    
    if (window.location.hostname === 'facebook.com') {
        document.querySelectorAll('[role="article"] [dir="auto"] span').forEach(el => {
            if (el.textContent.trim() && !el.querySelector('img, video')) {
                el.classList.add('cyberbullying-force-dark');
            }
        });
    }
    
    if (window.location.hostname === 'instagram.com') {
        document.querySelectorAll('header h2, [class*="author"]').forEach(el => {
            el.classList.add('cyberbullying-force-dark');
        });
    }
}
function restoreSensitiveElements(config) {
    // Remove platform-specific hiding
    if (config.hideSelectors) {
        config.hideSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                el.classList.remove('cyberbullying-force-dark');
            });
        });
    }
    
    // Remove nuclear darkening
    document.querySelectorAll('.cyberbullying-force-dark').forEach(el => {
        el.classList.remove('cyberbullying-force-dark');
    });
}
function clearHighlights() {
    document.querySelectorAll(`.${HIGHLIGHT_CLASS}`).forEach(el => {
        el.classList.remove(HIGHLIGHT_CLASS, HIGHLIGHT_BULLY_CLASS, HIGHLIGHT_SEVERE_CLASS, HIGHLIGHT_CLEAN_CLASS);
    });
}

async function safeAnalyzePage() {
    if (isScanning) return;
    isScanning = true;
    
    try {
        const { enabled } = await chrome.runtime.sendMessage({ action: 'getSettings' });
        if (!enabled) return;
        
        await analyzeVisibleContent();
    } catch (error) {
        console.debug('Page analysis error:', error);
    } finally {
        isScanning = false;
    }
}

async function analyzeVisibleContent() {
    const hostname = window.location.hostname;
    const config = PLATFORM_CONFIG[hostname] || PLATFORM_CONFIG.default;
    
    const { privacyMode } = await chrome.runtime.sendMessage({ action: 'getSettings' }) || { privacyMode: true };
    
    if (privacyMode !== false) {
        hideSensitiveElements(config);
    }
    
    try {
        detectionHistory.clear();
        detectedTexts = { clean: [], bully: [], severe: [] };
        currentCounts = { clean: 0, bully: 0, severe: 0 };
        
        const allElements = [];
        config.textContainers.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                if (!shouldIgnoreElement(el, config.ignoreSelectors)) {
                    allElements.push(el);
                }
            });
        });

        for (let i = 0; i < allElements.length; i += SCAN_CHUNK_SIZE) {
            const chunk = allElements.slice(i, i + SCAN_CHUNK_SIZE);
            await processElementChunk(chunk, config, hostname);
            await new Promise(resolve => setTimeout(resolve, SCAN_DELAY_MS));
        }
        
        updateCounts();
    } finally {
        if (privacyMode !== false) {
            restoreSensitiveElements(config);
        }
    }
}

function hideSensitiveElements(config) {
    if (!config.hideSelectors) return;
    
    config.hideSelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => {
            el.dataset.originalVisibility = el.style.visibility;
            el.dataset.originalOpacity = el.style.opacity;
            el.dataset.originalPosition = el.style.position;
            el.dataset.originalDisplay = el.style.display;
            
            el.style.visibility = 'hidden';
            el.style.opacity = '0';
            el.style.position = 'absolute';
            el.style.display = 'none';
            
            if (el.tagName === 'VIDEO') {
                el.pause();
                el.dataset.wasPlaying = !el.paused;
            }
        });
    });
}

function restoreSensitiveElements(config) {
    if (!config.hideSelectors) return;
    
    config.hideSelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => {
            el.style.visibility = el.dataset.originalVisibility || '';
            el.style.opacity = el.dataset.originalOpacity || '';
            el.style.position = el.dataset.originalPosition || '';
            el.style.display = el.dataset.originalDisplay || '';
            
            if (el.tagName === 'VIDEO' && el.dataset.wasPlaying === 'true') {
                el.play().catch(e => console.debug('Video play error:', e));
            }
            
            delete el.dataset.originalVisibility;
            delete el.dataset.originalOpacity;
            delete el.dataset.originalPosition;
            delete el.dataset.originalDisplay;
            delete el.dataset.wasPlaying;
        });
    });
}

function shouldIgnoreElement(el, ignoreSelectors) {
    if (analyzedElements.has(el)) return true;
    
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    const isVisible = (
        rect.width > 0 &&
        rect.height > 0 &&
        rect.bottom >= 0 &&
        rect.right >= 0 &&
        rect.top <= window.innerHeight &&
        rect.left <= window.innerWidth &&
        style.visibility === 'visible' &&
        style.display !== 'none' &&
        style.opacity !== '0'
    );
    
    return !isVisible || ignoreSelectors.some(selector => el.closest(selector));
}

async function processElementChunk(elements, config, hostname) {
    const { threshold } = await chrome.runtime.sendMessage({ action: 'getSettings' }) || { threshold: 0.7 };
    
    for (const el of elements) {
        if (analyzedElements.has(el)) continue;
        
        const text = el.textContent.trim();
        if (!isValidText(text, config)) continue;
        
        analyzedElements.add(el);
        
        try {
            const response = await chrome.runtime.sendMessage({
                action: 'detectCyberbullying',
                text
            });

            currentDetection = response;

            if (response?.status === 'success') {
                const { prediction, confidence } = response;
                if (confidence >= threshold) {
                    highlightElement(el, prediction, confidence, text);
                }
            }
        } catch (error) {
            console.debug('Element analysis error:', error);
        } finally {
            currentDetection = null;
        }
    }
}

function isValidText(text, config) {
    if (config.ignorePatterns.some(pattern => pattern.test(text))) {
        return false;
    }
    
    if (text.length < config.minTextLength || text.length > config.maxTextLength) {
        return false;
    }
    
    const words = text.split(/\s+/);
    const uniqueWords = new Set(words.map(w => w.toLowerCase()));
    if (uniqueWords.size < config.minUniqueWords) {
        return false;
    }
    
    if (/^[A-Z][a-z]+([A-Z][a-z]+)+$/.test(text)) {
        return false;
    }
    
    if (/^[A-Z\s]+$/.test(text)) {
        return false;
    }
    
    return true;
}

function hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash |= 0;
    }
    return hash;
}

async function highlightElement(element, severity, confidence, text) {
    if (element.querySelectorAll('img, video, iframe').length > 0) {
        return;
    }

    const normalizedText = text.toLowerCase().trim();
    const textHash = hashCode(normalizedText);
    
    if (detectionHistory.has(textHash)) {
        return;
    }
    
    detectionHistory.add(textHash);
    
    if (detectionHistory.size > DETECTION_HISTORY_SIZE) {
        const oldest = Array.from(detectionHistory).shift();
        detectionHistory.delete(oldest);
    }

    const { cleanColor, bullyColor, severeColor } = await chrome.storage.local.get([
        'cleanColor', 'bullyColor', 'severeColor'
    ]);

    const span = document.createElement('span');
    let highlightClass, highlightColor;
    
    if (severity === 0) {
        highlightClass = HIGHLIGHT_CLEAN_CLASS;
        highlightColor = cleanColor || '#28a745';
    } else if (severity === 1) {
        highlightClass = HIGHLIGHT_BULLY_CLASS;
        highlightColor = bullyColor || '#ffc107';
    } else {
        highlightClass = HIGHLIGHT_SEVERE_CLASS;
        highlightColor = severeColor || '#dc3545';
    }

    span.className = `${HIGHLIGHT_CLASS} ${highlightClass}`;
    span.title = severity === 0 ? 'Clean content' : 
                severity === 1 ? `Bullying (${(confidence * 100).toFixed(1)}% confidence)` : 
                `Severe bullying (${(confidence * 100).toFixed(1)}% confidence)`;
    span.dataset.cyberbullyingType = severity === 0 ? 'clean' : severity === 1 ? 'bully' : 'severe';
    span.dataset.detectionData = JSON.stringify({
        severity,
        confidence,
        text,
        explanations: currentDetection?.explanations || []
    });

    span.style.backgroundColor = `rgba(${hexToRgb(highlightColor)}, 0.3)`;
    span.style.boxShadow = `0 0 0 1px rgba(${hexToRgb(highlightColor)}, 0.5)`;

    const range = document.createRange();
    range.selectNodeContents(element);
    const fragment = range.extractContents();
    
    const walker = document.createTreeWalker(
        fragment,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode: function(node) {
                const text = node.nodeValue.trim();
                return text.length >= 5 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
            }
        },
        false
    );
    
    let node;
    const textNodes = [];
    while (node = walker.nextNode()) {
        textNodes.push(node);
    }
    
    textNodes.forEach(textNode => {
        const textSpan = span.cloneNode();
        textNode.parentNode.replaceChild(textSpan, textNode);
        textSpan.appendChild(textNode);
    });
    
    element.innerHTML = '';
    element.appendChild(fragment);
    
    element.querySelectorAll(`.${HIGHLIGHT_CLASS}`).forEach(hl => {
        hl.addEventListener('mouseenter', showExplanationTooltip);
        hl.addEventListener('mouseleave', hideExplanationTooltip);
        hl.addEventListener('click', handleHighlightClick);
    });
    
    const detection = { 
        content: text, 
        confidence, 
        element: span,
        explanations: currentDetection?.explanations || []
    };
    const targetArray = severity === 0 ? detectedTexts.clean : 
                      severity === 1 ? detectedTexts.bully : 
                      detectedTexts.severe;
    
    if (!targetArray.some(d => hashCode(d.content.toLowerCase().trim()) === textHash)) {
        targetArray.push(detection);
    }
}

function hexToRgb(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `${r}, ${g}, ${b}`;
}

function updateHighlightStyles(colors) {
    const style = document.createElement('style');
    style.id = 'cyberbullying-highlight-styles';
    style.textContent = `
        .${HIGHLIGHT_BULLY_CLASS} {
            background-color: rgba(${hexToRgb(colors.bully || '#ffc107')}, 0.3) !important;
            box-shadow: 0 0 0 1px rgba(${hexToRgb(colors.bully || '#ffc107')}, 0.5) !important;
        }
        .${HIGHLIGHT_SEVERE_CLASS} {
            background-color: rgba(${hexToRgb(colors.severe || '#dc3545')}, 0.3) !important;
            box-shadow: 0 0 0 1px rgba(${hexToRgb(colors.severe || '#dc3545')}, 0.5) !important;
        }
        .${HIGHLIGHT_CLEAN_CLASS} {
            background-color: rgba(${hexToRgb(colors.clean || '#28a745')}, 0.3) !important;
            box-shadow: 0 0 0 1px rgba(${hexToRgb(colors.clean || '#28a745')}, 0.5) !important;
        }
    `;
    
    const existingStyle = document.getElementById('cyberbullying-highlight-styles');
    if (existingStyle) {
        existingStyle.replaceWith(style);
    } else {
        document.head.appendChild(style);
    }
    
    document.querySelectorAll(`.${HIGHLIGHT_BULLY_CLASS}`).forEach(el => {
        el.style.backgroundColor = `rgba(${hexToRgb(colors.bully || '#ffc107')}, 0.3)`;
        el.style.boxShadow = `0 0 0 1px rgba(${hexToRgb(colors.bully || '#ffc107')}, 0.5)`;
    });
    
    document.querySelectorAll(`.${HIGHLIGHT_SEVERE_CLASS}`).forEach(el => {
        el.style.backgroundColor = `rgba(${hexToRgb(colors.severe || '#dc3545')}, 0.3)`;
        el.style.boxShadow = `0 0 0 1px rgba(${hexToRgb(colors.severe || '#dc3545')}, 0.5)`;
    });
    
    document.querySelectorAll(`.${HIGHLIGHT_CLEAN_CLASS}`).forEach(el => {
        el.style.backgroundColor = `rgba(${hexToRgb(colors.clean || '#28a745')}, 0.3)`;
        el.style.boxShadow = `0 0 0 1px rgba(${hexToRgb(colors.clean || '#28a745')}, 0.5)`;
    });
}

function showExplanationTooltip(event) {
    const element = event.currentTarget;
    const detectionData = JSON.parse(element.dataset.detectionData);
    
    const tooltip = document.createElement('div');
    tooltip.className = 'cyberbullying-explanation-tooltip';
    
    tooltip.innerHTML = `
        <div class="tooltip-header">
            <strong>${detectionData.severity === 0 ? '✅ Clean' : 
                     detectionData.severity === 1 ? '⚠️ Bullying' : 
                     '🚨 Severe Bullying'}</strong>
            <span>${(detectionData.confidence * 100).toFixed(1)}% confidence</span>
        </div>
        <div class="tooltip-content">
            <p class="detected-text">"${detectionData.text}"</p>
            <div class="explanations">
                <h4>Key Factors:</h4>
                ${detectionData.explanations.map(exp => `
                    <div class="explanation-item">
                        <span class="impact-direction ${exp.direction}">
                            ${exp.direction === 'positive' ? '↑' : '↓'}
                        </span>
                        <span class="explanation-text">${exp.description}</span>
                        <span class="impact-value">${Math.abs(exp.impact * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    const rect = element.getBoundingClientRect();
    tooltip.style.position = 'fixed';
    tooltip.style.left = `${rect.left + window.scrollX}px`;
    tooltip.style.top = `${rect.top + window.scrollY - 10}px`;
    tooltip.style.transform = 'translateY(-100%)';
    
    tooltip.id = 'cyberbullying-explanation-tooltip';
    document.body.appendChild(tooltip);
    
    const updatePosition = () => {
        const rect = element.getBoundingClientRect();
        tooltip.style.left = `${rect.left + window.scrollX}px`;
        tooltip.style.top = `${rect.top + window.scrollY - 10}px`;
    };
    
    window.addEventListener('scroll', updatePosition);
    element._tooltipListeners = { updatePosition };
}

function hideExplanationTooltip(event) {
    const tooltip = document.getElementById('cyberbullying-explanation-tooltip');
    if (tooltip) {
        tooltip.remove();
    }
    
    const element = event.currentTarget;
    if (element._tooltipListeners) {
        window.removeEventListener('scroll', element._tooltipListeners.updatePosition);
        delete element._tooltipListeners;
    }
}

function handleHighlightClick(event) {
    event.stopPropagation();
    const element = event.currentTarget;
    const type = element.dataset.cyberbullyingType;
    const text = element.textContent.trim();
    
    const popup = document.createElement('div');
    popup.className = 'cyberbullying-feedback-popup';
    popup.innerHTML = `
        <div class="feedback-title">Is this ${type === 'clean' ? 'clean' : 
                                      type === 'severe' ? 'severe bullying' : 'bullying'}?</div>
        <div class="feedback-buttons">
            <button class="feedback-yes">Yes</button>
            <button class="feedback-no">No</button>
            <button class="feedback-dismiss">×</button>
        </div>
    `;
    
    popup.querySelector('.feedback-yes').addEventListener('click', () => {
        sendUserFeedback(text, true, type);
        popup.remove();
    });
    
    popup.querySelector('.feedback-no').addEventListener('click', () => {
        sendUserFeedback(text, false, type);
        element.classList.remove(HIGHLIGHT_CLASS, HIGHLIGHT_BULLY_CLASS, HIGHLIGHT_SEVERE_CLASS, HIGHLIGHT_CLEAN_CLASS);
        element.title = '';
        popup.remove();
        updateCounts();
    });
    
    popup.querySelector('.feedback-dismiss').addEventListener('click', () => {
        popup.remove();
    });
    
    document.body.appendChild(popup);
    positionPopup(popup, element);
}

function positionPopup(popup, element) {
    const rect = element.getBoundingClientRect();
    popup.style.position = 'fixed';
    popup.style.left = `${rect.left + window.scrollX}px`;
    popup.style.top = `${rect.bottom + window.scrollY + 5}px`;
    
    setTimeout(() => {
        const clickOutsideHandler = (e) => {
            if (!popup.contains(e.target) && !element.contains(e.target)) {
                popup.remove();
                document.removeEventListener('click', clickOutsideHandler);
            }
        };
        document.addEventListener('click', clickOutsideHandler);
    }, 0);
}

function sendUserFeedback(text, isCorrect, type) {
    chrome.runtime.sendMessage({
        action: 'logFeedback',
        text,
        isCorrect,
        type
    }).catch(() => {});
}

function updateCounts() {
    const cleanCount = document.querySelectorAll(`.${HIGHLIGHT_CLEAN_CLASS}`).length;
    const bullyCount = document.querySelectorAll(`.${HIGHLIGHT_BULLY_CLASS}`).length;
    const severeCount = document.querySelectorAll(`.${HIGHLIGHT_SEVERE_CLASS}`).length;
    
    currentCounts = { clean: cleanCount, bully: bullyCount, severe: severeCount };
    updateCountBadge(currentCounts);
    
    chrome.runtime.sendMessage({
        action: 'updateCounts',
        counts: currentCounts
    }).catch(() => {});
}

function updateCountBadge(counts) {
    const total = counts.bully + counts.severe;
    chrome.runtime.sendMessage({
        action: 'setBadge',
        text: total > 0 ? total.toString() : '',
        color: counts.severe > 0 ? '#d32f2f' : '#ff9800'
    }).catch(() => {});
}

function setupScrollAnalysis() {
    if (scrollHandler) window.removeEventListener('scroll', scrollHandler);
    
    let scrollTimeout;
    scrollHandler = () => {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(() => {
            if (!isScanning) safeAnalyzePage();
        }, 500);
    };
    
    window.addEventListener('scroll', scrollHandler);
}

function setupMutationObserver() {
    if (mutationObserver) mutationObserver.disconnect();
    
    mutationObserver = new MutationObserver(mutations => {
        const hasAddedNodes = mutations.some(mutation => 
            mutation.addedNodes.length > 0 &&
            Array.from(mutation.addedNodes).some(node => 
                node.nodeType === Node.ELEMENT_NODE)
        );
        
        if (hasAddedNodes && !isScanning) {
            safeAnalyzePage();
        }
    });
    
    mutationObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: false,
        characterData: false
    });
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const handleRequest = async () => {
        try {
            if (request.action === 'reanalyzePage') {
                clearHighlights();
                analyzedElements.clear();
                await safeAnalyzePage();
                return { status: 'success' };
            }
            else if (request.action === 'getDetectionCounts') {
                return {
                    clean: detectedTexts.clean.length,
                    bully: detectedTexts.bully.length,
                    severe: detectedTexts.severe.length
                };
            }
            else if (request.action === 'getDetectedTexts') {
                return { texts: detectedTexts[request.type] || [] };
            }
        } catch (error) {
            console.debug('Content script error:', error);
            return { error: error.message, status: 'error' };
        }
    };
    
    handleRequest().then(sendResponse);
    return true;
});

const styleElement = document.createElement('style');
styleElement.textContent = `
.${HIGHLIGHT_CLASS} {
    position: relative;
    display: inline;
    padding: 0 !important;
    margin: 0 !important;
    line-height: inherit !important;
    border-radius: 3px;
    transition: background-color 0.3s ease;
}

.${HIGHLIGHT_BULLY_CLASS} {
    background-color: rgba(255, 152, 0, 0.3);
    box-shadow: 0 0 0 1px rgba(255, 152, 0, 0.5);
}

.${HIGHLIGHT_SEVERE_CLASS} {
    background-color: rgba(211, 47, 47, 0.3);
    box-shadow: 0 0 0 1px rgba(211, 47, 47, 0.5);
}

.${HIGHLIGHT_CLEAN_CLASS} {
    background-color: rgba(40, 167, 69, 0.3);
    box-shadow: 0 0 0 1px rgba(40, 167, 69, 0.5);
}

.cyberbullying-feedback-popup {
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    z-index: 99999;
    font-family: Arial, sans-serif;
    font-size: 14px;
    max-width: 250px;
}

.feedback-title {
    margin-bottom: 8px;
    font-weight: bold;
}

.feedback-buttons {
    display: flex;
    gap: 5px;
}

.feedback-buttons button {
    padding: 4px 8px;
    border: 1px solid #ddd;
    border-radius: 3px;
    cursor: pointer;
    font-size: 12px;
}

.feedback-buttons .feedback-yes {
    background-color: #4caf50;
    color: white;
    border-color: #4caf50;
}

.feedback-buttons .feedback-no {
    background-color: #f44336;
    color: white;
    border-color: #f44336;
}

.feedback-buttons .feedback-dismiss {
    margin-left: auto;
    background: none;
    border: none;
    font-size: 16px;
    line-height: 1;
    padding: 0 5px;
}

.cyberbullying-explanation-tooltip {
    position: absolute;
    z-index: 999999;
    background: white;
    border-radius: 6px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    padding: 12px;
    max-width: 350px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    font-size: 13px;
    line-height: 1.4;
    color: #333;
    pointer-events: none;
}

.tooltip-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #eee;
}

.tooltip-header strong {
    font-size: 14px;
    color: #d32f2f;
}

.tooltip-header span {
    font-size: 12px;
    color: #666;
    background: #f5f5f5;
    padding: 2px 6px;
    border-radius: 10px;
}

.detected-text {
    font-style: italic;
    color: #555;
    margin: 8px 0;
    padding: 8px;
    background: #f9f9f9;
    border-radius: 4px;
    font-size: 12px;
    line-height: 1.5;
}

.explanations {
    margin-top: 10px;
}

.explanations h4 {
    margin: 0 0 8px 0;
    font-size: 13px;
    color: #333;
}

.explanation-item {
    display: flex;
    align-items: center;
    margin-bottom: 6px;
    padding: 4px 0;
}

.impact-direction {
    display: inline-block;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    text-align: center;
    line-height: 20px;
    margin-right: 8px;
    font-weight: bold;
}

.impact-direction.positive {
    background-color: rgba(211, 47, 47, 0.1);
    color: #d32f2f;
}

.impact-direction.negative {
    background-color: rgba(40, 167, 69, 0.1);
    color: #28a745;
}

.explanation-text {
    flex: 1;
    font-size: 12px;
}

.impact-value {
    font-size: 11px;
    color: #666;
    margin-left: 8px;
    font-family: monospace;
}
`;
document.head.appendChild(styleElement);

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initContentScript);
} else {
    initContentScript();
}