document.addEventListener('DOMContentLoaded', function() {
    const elements = {
        toggleEnabled: document.getElementById('toggleEnabled'),
        statusText: document.getElementById('statusText'),
        thresholdSlider: document.getElementById('threshold'),
        thresholdValue: document.getElementById('thresholdValue'),
        cleanCount: document.getElementById('cleanCount'),
        bullyCount: document.getElementById('bullyCount'),
        severeCount: document.getElementById('severeCount'),
        rescanBtn: document.getElementById('rescanBtn'),
        reportBtn: document.getElementById('reportBtn'),
        apiStatus: document.getElementById('apiStatus'),
        manualText: document.getElementById('manualText'),
        testTextBtn: document.getElementById('testTextBtn'),
        manualResult: document.getElementById('manualResult'),
        settingsBtn: document.getElementById('settingsBtn'),
        countsLastUpdated: document.getElementById('countsLastUpdated'),
        bullyStatCard: document.getElementById('bullyStatCard'),
        severeStatCard: document.getElementById('severeStatCard'),
        detectionsModal: document.getElementById('detectionsModal'),
        modalClose: document.querySelector('.close'),
        bullyTab: document.getElementById('bullyTab'),
        severeTab: document.getElementById('severeTab'),
        bullyList: document.getElementById('bullyList'),
        severeList: document.getElementById('severeList'),
        tabButtons: document.querySelectorAll('.tab-btn'),
        cleanColor: document.getElementById('cleanColor'),
        bullyColor: document.getElementById('bullyColor'),
        severeColor: document.getElementById('severeColor'),
        saveColorSettings: document.getElementById('saveColorSettings'),
        settingsTab: document.getElementById('settingsTab'),
        privacyToggle: document.getElementById('privacyToggle')
    };

    let popupOpen = true;
    let activeTabId = null;
    let countsUpdateInterval = null;

    async function initPopup() {
        await getActiveTab();
        await loadSettings();
        await checkApiStatus();
        await updateDetectionCounts();
        setupEventListeners();
        
        countsUpdateInterval = setInterval(updateDetectionCounts, 3000);
        keepAlive();
    }

    function keepAlive() {
        let keepAliveInterval = setInterval(() => {
            if (!popupOpen) {
                clearInterval(keepAliveInterval);
                return;
            }
            chrome.runtime.sendMessage({ action: 'keepAlive' });
        }, 10000);
        
        window.addEventListener('unload', () => {
            popupOpen = false;
            if (countsUpdateInterval) clearInterval(countsUpdateInterval);
        });
    }

    async function getActiveTab() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            activeTabId = tab?.id || null;
        } catch (error) {
            console.error('Error getting active tab:', error);
        }
    }

    async function loadSettings() {
        try {
            const result = await chrome.runtime.sendMessage({ action: 'getSettings' });
            if (result) {
                elements.toggleEnabled.checked = result.enabled !== false;
                updateStatusText();

                const threshold = result.threshold || 0.7;
                elements.thresholdSlider.value = threshold;
                elements.thresholdValue.textContent = `${Math.round(threshold * 100)}%`;

                elements.cleanColor.value = result.cleanColor || '#28a745';
                elements.bullyColor.value = result.bullyColor || '#ffc107';
                elements.severeColor.value = result.severeColor || '#dc3545';
                
                elements.privacyToggle.checked = result.privacyMode !== false;
            }
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    async function updateDetectionCounts() {
        try {
            if (!activeTabId) return;
            
            const response = await chrome.tabs.sendMessage(activeTabId, { action: 'getDetectionCounts' });
            if (response) {
                elements.bullyCount.textContent = response.bully || '0';
                elements.severeCount.textContent = response.severe || '0';
                elements.cleanCount.textContent = response.clean || '0';
                
                if (elements.countsLastUpdated) {
                    elements.countsLastUpdated.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
                }
            } else {
                elements.cleanCount.textContent = 'N/A';
                elements.bullyCount.textContent = 'N/A';
                elements.severeCount.textContent = 'N/A';
                if (elements.countsLastUpdated) {
                    elements.countsLastUpdated.textContent = 'Counts not available';
                }
            }
        } catch (error) {
            console.error('Error getting detection counts:', error);
            elements.cleanCount.textContent = 'N/A';
            elements.bullyCount.textContent = 'N/A';
            elements.severeCount.textContent = 'N/A';
            if (elements.countsLastUpdated) {
                elements.countsLastUpdated.textContent = 'Error updating counts';
            }
        }
    }

    async function checkApiStatus() {
        try {
            const response = await chrome.runtime.sendMessage({ action: 'checkApiStatus' });
            if (response) {
                updateApiStatusUI(response);
            }
        } catch (error) {
            console.error('Error checking API status:', error);
            updateApiStatusUI({ status: 'disconnected', error: error.message });
        }
    }

    function updateApiStatusUI(response) {
        const statusDot = elements.apiStatus.querySelector('.status-dot');
        const statusTextElement = elements.apiStatus.querySelector('.status-text');

        if (response.status === 'connected') {
            statusDot.classList.add('connected');
            statusDot.classList.remove('disconnected');
            statusTextElement.textContent = 'API Connected';
            statusTextElement.title = `Using ${response.apiUrl}`;
        } else {
            statusDot.classList.add('disconnected');
            statusDot.classList.remove('connected');
            statusTextElement.textContent = 'API Disconnected';
            statusTextElement.title = response.error || 'API connection failed';
        }
    }

    function updateStatusText() {
        elements.statusText.textContent = elements.toggleEnabled.checked ? 'Enabled' : 'Disabled';
        elements.statusText.style.color = elements.toggleEnabled.checked ? 'var(--success-color)' : 'var(--danger-color)';
    }

    async function showDetections(type) {
        try {
            if (!activeTabId) return;
            
            const response = await chrome.tabs.sendMessage(activeTabId, { 
                action: 'getDetectedTexts',
                type
            });
            
            if (response) {
                const listElement = type === 'bully' ? elements.bullyList : elements.severeList;
                listElement.innerHTML = '';
                
                response.texts.forEach(text => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${text.content}</td>
                        <td>${Math.round(text.confidence * 100)}%</td>
                    `;
                    listElement.appendChild(row);
                });
                
                elements.detectionsModal.style.display = 'block';
            }
        } catch (error) {
            console.error('Error getting detected texts:', error);
        }
    }

    function setupEventListeners() {
        elements.toggleEnabled.addEventListener('change', async function() {
            const enabled = this.checked;
            try {
                await chrome.runtime.sendMessage({
                    action: 'saveSettings',
                    enabled,
                    threshold: parseFloat(elements.thresholdSlider.value),
                    cleanColor: elements.cleanColor.value,
                    bullyColor: elements.bullyColor.value,
                    severeColor: elements.severeColor.value,
                    privacyMode: elements.privacyToggle.checked
                });
                updateStatusText();
                
                if (activeTabId) {
                    chrome.tabs.sendMessage(activeTabId, {
                        action: 'updateSettings',
                        enabled,
                        threshold: parseFloat(elements.thresholdSlider.value),
                        privacyMode: elements.privacyToggle.checked,
                        colors: {
                            clean: elements.cleanColor.value,
                            bully: elements.bullyColor.value,
                            severe: elements.severeColor.value
                        }
                    }).catch(() => {});
                }
                
                updateDetectionCounts();
            } catch (error) {
                console.error('Error saving settings:', error);
            }
        });

        elements.thresholdSlider.addEventListener('input', function() {
            const threshold = parseFloat(this.value);
            elements.thresholdValue.textContent = `${Math.round(threshold * 100)}%`;
        });

        elements.thresholdSlider.addEventListener('change', async function() {
            const threshold = parseFloat(this.value);
            try {
                await chrome.runtime.sendMessage({
                    action: 'saveSettings',
                    enabled: elements.toggleEnabled.checked,
                    threshold,
                    cleanColor: elements.cleanColor.value,
                    bullyColor: elements.bullyColor.value,
                    severeColor: elements.severeColor.value,
                    privacyMode: elements.privacyToggle.checked
                });
                
                if (activeTabId) {
                    chrome.tabs.sendMessage(activeTabId, {
                        action: 'updateSettings',
                        enabled: elements.toggleEnabled.checked,
                        threshold,
                        privacyMode: elements.privacyToggle.checked,
                        colors: {
                            clean: elements.cleanColor.value,
                            bully: elements.bullyColor.value,
                            severe: elements.severeColor.value
                        }
                    }).catch(() => {});
                }
                
                updateDetectionCounts();
            } catch (error) {
                console.error('Error saving threshold:', error);
            }
        });

        elements.privacyToggle.addEventListener('change', async function() {
            try {
                await chrome.runtime.sendMessage({
                    action: 'saveSettings',
                    enabled: elements.toggleEnabled.checked,
                    threshold: parseFloat(elements.thresholdSlider.value),
                    cleanColor: elements.cleanColor.value,
                    bullyColor: elements.bullyColor.value,
                    severeColor: elements.severeColor.value,
                    privacyMode: this.checked
                });
                
                if (activeTabId) {
                    chrome.tabs.sendMessage(activeTabId, {
                        action: 'updateSettings',
                        enabled: elements.toggleEnabled.checked,
                        threshold: parseFloat(elements.thresholdSlider.value),
                        privacyMode: this.checked,
                        colors: {
                            clean: elements.cleanColor.value,
                            bully: elements.bullyColor.value,
                            severe: elements.severeColor.value
                        }
                    }).catch(() => {});
                }
            } catch (error) {
                console.error('Error saving privacy mode:', error);
            }
        });

        elements.rescanBtn.addEventListener('click', async function() {
            if (!activeTabId) return;
            
            try {
                elements.rescanBtn.textContent = 'Scanning...';
                elements.rescanBtn.disabled = true;
                
                const response = await chrome.tabs.sendMessage(activeTabId, { action: 'reanalyzePage' });
                if (response?.status === 'success') {
                    await updateDetectionCounts();
                }
            } catch (error) {
                console.error('Error rescanning page:', error);
            } finally {
                elements.rescanBtn.textContent = 'Rescan Page';
                elements.rescanBtn.disabled = false;
            }
        });

        elements.reportBtn.addEventListener('click', function() {
            chrome.tabs.create({ url: 'x' });
        });

        elements.settingsBtn.addEventListener('click', function() {
            elements.detectionsModal.style.display = 'block';
        });

        elements.bullyStatCard.addEventListener('click', () => showDetections('bully'));
        elements.severeStatCard.addEventListener('click', () => showDetections('severe'));

        elements.modalClose.addEventListener('click', () => {
            elements.detectionsModal.style.display = 'none';
        });

        window.addEventListener('click', (event) => {
            if (event.target === elements.detectionsModal) {
                elements.detectionsModal.style.display = 'none';
            }
        });

        elements.tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.dataset.tab;
                
                elements.tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.getElementById(`${tabName}Tab`).classList.add('active');
            });
        });

        elements.saveColorSettings.addEventListener('click', async function() {
            try {
                await chrome.runtime.sendMessage({
                    action: 'saveSettings',
                    enabled: elements.toggleEnabled.checked,
                    threshold: parseFloat(elements.thresholdSlider.value),
                    cleanColor: elements.cleanColor.value,
                    bullyColor: elements.bullyColor.value,
                    severeColor: elements.severeColor.value,
                    privacyMode: elements.privacyToggle.checked
                });
                
                if (activeTabId) {
                    chrome.tabs.sendMessage(activeTabId, {
                        action: 'updateHighlightColors',
                        colors: {
                            clean: elements.cleanColor.value,
                            bully: elements.bullyColor.value,
                            severe: elements.severeColor.value
                        }
                    }).catch(() => {});
                }
                
                const originalText = this.textContent;
                this.textContent = 'Saved!';
                setTimeout(() => {
                    this.textContent = originalText;
                }, 2000);
            } catch (error) {
                console.error('Error saving colors:', error);
            }
        });

        elements.testTextBtn.addEventListener('click', async function() {
            const text = elements.manualText.value.trim();
            if (!text) {
                showManualResult("❗ Please enter some text to analyze.", 'error');
                return;
            }

            showManualResult("⏳ Analyzing text...", 'loading');
            elements.testTextBtn.disabled = true;
            
            try {
                const response = await chrome.runtime.sendMessage({
                    action: 'analyzeText',
                    payload: text
                });

                if (!response) {
                    showManualResult("⚠️ Failed to analyze text.", 'error');
                    return;
                }

                if (response.status === 'error') {
                    showManualResult(`⚠️ Error: ${response.error || 'Unknown error'}`, 'error');
                } else {
                    const { prediction, confidence, probabilities } = response;
                    
                    let resultText, resultClass;
                    if (prediction === 0) {
                        resultText = "✅ Clean - No cyberbullying detected";
                        resultClass = 'clean';
                    } else if (prediction === 1) {
                        resultText = `⚠️ Bullying Detected (${Math.round(confidence * 100)}% confidence)`;
                        resultClass = 'bully';
                    } else if (prediction === 2) {
                        resultText = `🚨 Severe Bullying (${Math.round(confidence * 100)}% confidence)`;
                        resultClass = 'severe';
                    }
                    
                    if (probabilities) {
                        resultText += `<div class="probability-bar">
                            <div class="probability clean" style="width: ${probabilities[0] * 100}%"></div>
                            <div class="probability bully" style="width: ${probabilities[1] * 100}%"></div>
                            <div class="probability severe" style="width: ${probabilities[2] * 100}%"></div>
                        </div>
                        <div class="probability-labels">
                            <span>Clean: ${Math.round(probabilities[0] * 100)}%</span>
                            <span>Bully: ${Math.round(probabilities[1] * 100)}%</span>
                            <span>Severe: ${Math.round(probabilities[2] * 100)}%</span>
                        </div>`;
                    }
                    
                    showManualResult(resultText, resultClass);
                }
            } catch (error) {
                console.error('Manual detection error:', error);
                showManualResult("⚠️ Failed to analyze text.", 'error');
            } finally {
                elements.testTextBtn.disabled = false;
            }
        });
    }

    function showManualResult(content, type = 'info') {
        if (!elements.manualResult) return;
        
        elements.manualResult.innerHTML = content;
        elements.manualResult.className = 'result-container';
        
        switch (type) {
            case 'clean':
                elements.manualResult.classList.add('result-clean');
                break;
            case 'bully':
                elements.manualResult.classList.add('result-bully');
                break;
            case 'severe':
                elements.manualResult.classList.add('result-severe');
                break;
            case 'error':
                elements.manualResult.classList.add('result-error');
                break;
            case 'loading':
                elements.manualResult.classList.add('result-loading');
                break;
        }
    }

    initPopup();
});