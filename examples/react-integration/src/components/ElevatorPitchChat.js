import React, { useState, useEffect, useRef } from 'react';
import { CreateMLCEngine } from '@mlc-ai/web-llm';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import LoadingStatus from './LoadingStatus';
import './ElevatorPitchChat.css';

const ElevatorPitchChat = () => {
  const [engine, setEngine] = useState(null);
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: '👋 Hello! I\'m your personalized elevator pitch assistant. I can help you craft compelling professional introductions. Try asking me about elevator pitches, professional backgrounds, or personal branding!',
      timestamp: new Date()
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('initializing');
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Initialize WebLLM engine
  useEffect(() => {
    initializeEngine();
  }, []);

  const initializeEngine = async () => {
    try {
      setLoadingStatus('initializing');
      setLoadingProgress(0);

      // Configuration for your fine-tuned model
      const appConfig = {
        model_list: [
          {
            // Replace with your HuggingFace model URL
            model: "https://huggingface.co/your-username/PersonalizedQwen-0.6B-Chat-q4f16_1-MLC",
            model_id: "PersonalizedQwen-0.6B-Chat-q4f16_1-MLC",
            // Replace with your compiled model library URL
            model_lib: "https://github.com/your-username/your-repo/releases/download/v1.0/PersonalizedQwen-0.6B-Chat-q4f16_1-MLC-webgpu.wasm",
            required_features: ["shader-f16"],
            overrides: {
              context_window_size: 2048,
              prefill_chunk_size: 512
            }
          }
        ]
      };

      // Fallback to demo model for now
      const selectedModel = "Llama-3-8B-Instruct-q4f32_1-MLC";

      setLoadingStatus('loading');
      
      const engineInstance = await CreateMLCEngine(
        selectedModel, // Replace with your model ID when ready
        {
          // appConfig: appConfig, // Uncomment when using your custom model
          initProgressCallback: (progress) => {
            setLoadingProgress(Math.round(progress.progress * 100));
          }
        }
      );

      setEngine(engineInstance);
      setIsInitialized(true);
      setLoadingStatus('ready');
    } catch (error) {
      console.error('Failed to initialize engine:', error);
      setLoadingStatus('error');
    }
  };

  const sendMessage = async (content) => {
    if (!engine || !content.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setIsTyping(true);

    try {
      // Prepare conversation context
      const conversationHistory = [
        {
          role: "system",
          content: "You are a professional career coach and elevator pitch expert. Help users create compelling, personalized elevator pitches and professional introductions. Be concise, engaging, and focus on unique value propositions. Ask follow-up questions when needed to personalize responses."
        },
        ...messages.slice(-10).map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        {
          role: "user",
          content: content
        }
      ];

      // Generate response
      const response = await engine.chat.completions.create({
        messages: conversationHistory,
        temperature: 0.7,
        max_tokens: 512,
        stream: false
      });

      const aiResponse = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.choices[0].message.content,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setIsTyping(false);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: 1,
        role: 'assistant',
        content: '👋 Hello! I\'m your personalized elevator pitch assistant. How can I help you create a compelling professional introduction today?',
        timestamp: new Date()
      }
    ]);
  };

  const quickPrompts = [
    "Tell me about yourself",
    "What's your elevator pitch?",
    "Describe your background",
    "What makes you unique?",
    "How do I introduce myself professionally?",
    "What should I highlight in my pitch?"
  ];

  return (
    <div className="elevator-pitch-chat">
      {!isInitialized && (
        <LoadingStatus 
          status={loadingStatus} 
          progress={loadingProgress}
          onRetry={initializeEngine}
        />
      )}
      
      {isInitialized && (
        <>
          <div className="chat-header">
            <div className="chat-title">
              <h2>Elevator Pitch Assistant</h2>
              <span className="status-indicator">🟢 Ready</span>
            </div>
            <button 
              className="clear-button"
              onClick={clearChat}
              title="Clear conversation"
            >
              🗑️
            </button>
          </div>

          <div className="chat-messages">
            <MessageList 
              messages={messages} 
              isTyping={isTyping}
            />
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-container">
            <div className="quick-prompts">
              {quickPrompts.map((prompt, index) => (
                <button
                  key={index}
                  className="quick-prompt-button"
                  onClick={() => sendMessage(prompt)}
                  disabled={isLoading}
                >
                  {prompt}
                </button>
              ))}
            </div>
            
            <MessageInput 
              onSendMessage={sendMessage}
              disabled={isLoading}
              placeholder="Ask me about elevator pitches..."
            />
          </div>
        </>
      )}
    </div>
  );
};

export default ElevatorPitchChat;