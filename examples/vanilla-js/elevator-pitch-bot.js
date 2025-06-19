// Elevator Pitch Bot - WebLLM Integration
class ElevatorPitchBot {
    constructor() {
        this.engine = null;
        this.isLoading = false;
        this.isInitialized = false;
        this.messages = [];
        this.retryCount = 0;
        this.maxRetries = 3;
        
        // DOM elements
        this.statusEl = document.getElementById('status');
        this.chatContainer = document.getElementById('chatContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        this.init();
    }

    async checkPrerequisites() {
        // Check if WebLLM is available
        if (typeof webllm === 'undefined') {
            throw new Error('WebLLM library is not available. Please check your internet connection.');
        }
        
        if (!webllm || !webllm.CreateMLCEngine) {
            throw new Error('WebLLM library not loaded properly. Please refresh the page.');
        }
        
        // Check internet connection
        if (!navigator.onLine) {
            throw new Error('No internet connection detected. Please check your connection.');
        }
        
        // Check secure context
        if (!window.isSecureContext) {
            throw new Error('Requires HTTPS or localhost for security.');
        }
        
        console.log('Prerequisites check passed');
    }

    async loadWebLLMWithRetry() {
        const TIMEOUT_MS = 300000; // 5 minutes timeout
        
        while (this.retryCount < this.maxRetries) {
            try {
                this.updateStatus('loading', `Loading AI model... (Attempt ${this.retryCount + 1}/${this.maxRetries})`);
                
                // Create timeout promise
                const timeoutPromise = new Promise((_, reject) => {
                    setTimeout(() => {
                        reject(new Error(`Model loading timed out after ${TIMEOUT_MS / 1000} seconds`));
                    }, TIMEOUT_MS);
                });
                
                // Create engine loading promise
                const enginePromise = webllm.CreateMLCEngine(
                    "Llama-3.2-1B-Instruct-q4f16_1-MLC", // Use the same model as working implementation
                    {
                        initProgressCallback: (report) => {
                            console.log('Loading progress:', report);
                            if (report.progress !== undefined) {
                                const progress = Math.round(report.progress * 100);
                                this.updateStatus('loading', `Loading model: ${progress}% - ${report.text || 'Downloading...'}`);
                            } else if (report.text) {
                                this.updateStatus('loading', `${report.text}`);
                            }
                        }
                    }
                );
                
                // Race between engine loading and timeout
                this.engine = await Promise.race([enginePromise, timeoutPromise]);
                
                console.log('WebLLM engine loaded successfully');
                return true;
                
            } catch (error) {
                console.error(`Loading attempt ${this.retryCount + 1} failed:`, error);
                this.retryCount++;
                
                if (this.retryCount >= this.maxRetries) {
                    throw new Error(`Failed to load model after ${this.maxRetries} attempts: ${error.message}`);
                }
                
                // Exponential backoff
                const waitTime = Math.min(5000 * Math.pow(2, this.retryCount - 1), 30000);
                this.updateStatus('loading', `Retrying in ${waitTime/1000} seconds...`);
                await new Promise(resolve => setTimeout(resolve, waitTime));
            }
        }
    }

    async init() {
        try {
            await this.initializeModel();
            this.setupEventListeners();
        } catch (error) {
            console.error('Failed to initialize bot:', error);
            this.updateStatus('error', 'Failed to load model. Please refresh and try again.');
        }
    }

    async initializeModel() {
        try {
            this.updateStatus('loading', 'Checking prerequisites...');
            await this.checkPrerequisites();
            
            this.updateStatus('loading', 'Initializing AI model...');
            console.log('WebLLM loaded successfully:', webllm);
            
            // Load WebLLM with retry mechanism (like the working implementation)
            await this.loadWebLLMWithRetry();
            
            this.isInitialized = true;
            this.updateStatus('ready', '✅ Model loaded successfully! Ready to chat.');
            this.enableInput();
            
        } catch (error) {
            console.error('Model initialization failed:', error);
            
            // Provide user-friendly error messages
            let errorMessage = 'Failed to load AI model. ';
            
            if (error.message.includes('network') || error.message.includes('connection')) {
                errorMessage += 'Please check your internet connection and try again.';
            } else if (error.message.includes('timeout')) {
                errorMessage += 'Loading timed out. Please try again or check your connection.';
            } else if (error.message.includes('WebLLM')) {
                errorMessage += 'AI library failed to load. Please refresh the page.';
            } else {
                errorMessage += 'Please refresh the page and try again.';
            }
            
            this.updateStatus('error', errorMessage);
            throw error;
        }
    }

    setupEventListeners() {
        this.messageInput.addEventListener('input', () => {
            this.sendButton.disabled = !this.messageInput.value.trim() || this.isLoading;
        });
    }

    updateStatus(type, message) {
        this.statusEl.className = `status ${type}`;
        this.statusEl.innerHTML = message;
    }

    enableInput() {
        this.sendButton.disabled = false;
        this.messageInput.disabled = false;
        this.messageInput.focus();
    }

    disableInput() {
        this.sendButton.disabled = true;
        this.messageInput.disabled = true;
    }

    showTypingIndicator() {
        this.typingIndicator.classList.add('show');
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    hideTypingIndicator() {
        this.typingIndicator.classList.remove('show');
    }

    addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = isUser ? 'You' : 'AI';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (isUser) {
            messageContent.textContent = content;
        } else {
            // Format AI response with better typography
            messageContent.innerHTML = this.formatResponse(content);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        // Insert before typing indicator
        this.chatContainer.insertBefore(messageDiv, this.typingIndicator);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        
        // Store message
        this.messages.push({
            role: isUser ? 'user' : 'assistant',
            content: content
        });
    }

    formatResponse(content) {
        // Basic formatting for better readability
        return content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    async sendMessage(message = null) {
        if (!this.isInitialized || this.isLoading) return;
        
        const userMessage = message || this.messageInput.value.trim();
        if (!userMessage) return;
        
        // Clear input
        this.messageInput.value = '';
        this.isLoading = true;
        this.disableInput();
        
        // Add user message
        this.addMessage(userMessage, true);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            // Prepare conversation context (similar to working implementation)
            const conversationHistory = [
                {
                    role: "system",
                    content: "You are a professional career coach and elevator pitch expert specializing in helping people create compelling personal introductions. You help users craft personalized elevator pitches that highlight their unique value propositions, skills, and experiences. Be engaging, ask relevant follow-up questions, and provide actionable advice. Keep responses concise but helpful."
                },
                ...this.messages.slice(-8), // Keep last 8 messages for context
                {
                    role: "user",
                    content: userMessage
                }
            ];
            
            // Generate response using the same parameters as working implementation
            const response = await this.engine.chat.completions.create({
                messages: conversationHistory,
                temperature: 0.7,
                max_tokens: 400, // Match working implementation
                stream: false
            });
            
            const aiResponse = response.choices[0].message.content;
            
            // Hide typing indicator and add AI response
            this.hideTypingIndicator();
            this.addMessage(aiResponse, false);
            
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            
            // Provide specific error messages based on error type
            let errorMessage = 'Sorry, I encountered an error. ';
            if (error.message.includes('network') || error.message.includes('fetch')) {
                errorMessage += 'Please check your internet connection and try again.';
            } else if (error.message.includes('timeout')) {
                errorMessage += 'The request timed out. Please try again.';
            } else {
                errorMessage += 'Please try again in a moment.';
            }
            
            this.addMessage(errorMessage, false);
        } finally {
            this.isLoading = false;
            this.enableInput();
        }
    }

    clearChat() {
        const messages = this.chatContainer.querySelectorAll('.message');
        messages.forEach((msg, index) => {
            if (index > 0) { // Keep the welcome message
                msg.remove();
            }
        });
        this.messages = [];
    }
}

// Global functions for HTML event handlers
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function sendMessage() {
    if (window.pitchBot) {
        window.pitchBot.sendMessage();
    }
}

function sendQuickMessage(message) {
    if (window.pitchBot) {
        window.pitchBot.sendMessage(message);
    }
}

function clearChat() {
    if (window.pitchBot) {
        window.pitchBot.clearChat();
    }
}

// Initialize the bot when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.pitchBot = new ElevatorPitchBot();
    });
} else {
    // DOM is already loaded
    window.pitchBot = new ElevatorPitchBot();
}

// Handle page visibility change to pause/resume model
document.addEventListener('visibilitychange', () => {
    if (window.pitchBot && window.pitchBot.engine) {
        if (document.hidden) {
            // Page is hidden, you could implement model pause here if needed
            console.log('Page hidden - model continues running');
        } else {
            // Page is visible again
            console.log('Page visible - model ready');
        }
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ElevatorPitchBot;
}