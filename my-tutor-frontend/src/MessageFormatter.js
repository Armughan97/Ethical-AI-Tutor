import React, { useState, useEffect, useRef } from 'react';

// Component to format message content
const FormattedMessage = ({ content }) => {
  // Function to parse and format the message
  const formatMessage = (text) => {
    // Replace **text** with bold formatting
    let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Replace numbered lists (1. text) with proper formatting
    formatted = formatted.replace(/(\d+\.\s\*\*[^*]+\*\*[^0-9]*)/g, '<div class="mb-2">$1</div>');
    
    // Add line breaks for better readability
    formatted = formatted.replace(/(\d+\.\s)/g, '<br/>$1');
    
    // Clean up multiple line breaks
    formatted = formatted.replace(/(<br\/>){2,}/g, '<br/><br/>');
    
    return formatted;
  };

  return (
    <div 
      className="formatted-message"
      dangerouslySetInnerHTML={{ __html: formatMessage(content) }}
    />
  );
};

// Alternative component using manual parsing (safer than dangerouslySetInnerHTML)
const SafeFormattedMessage = ({ content }) => {
  const parseMessage = (text) => {
    const parts = [];
    let currentIndex = 0;
    
    // Split by numbered items first
    const numberedSections = text.split(/(\d+\.\s\*\*[^*]+\*\*)/);
    
    return numberedSections.map((section, index) => {
      if (section.match(/^\d+\.\s\*\*/)) {
        // This is a numbered item with bold text
        const parts = section.split(/(\*\*[^*]+\*\*)/);
        return (
          <div key={index} className="mb-3">
            {parts.map((part, partIndex) => {
              if (part.match(/^\*\*.*\*\*$/)) {
                // Bold text
                return <strong key={partIndex}>{part.replace(/\*\*/g, '')}</strong>;
              }
              return part;
            })}
          </div>
        );
      } else {
        // Regular text that might contain bold formatting
        const parts = section.split(/(\*\*[^*]+\*\*)/);
        return (
          <span key={index}>
            {parts.map((part, partIndex) => {
              if (part.match(/^\*\*.*\*\*$/)) {
                return <strong key={partIndex}>{part.replace(/\*\*/g, '')}</strong>;
              }
              return part.split('\n').map((line, lineIndex, array) => (
                <React.Fragment key={`${partIndex}-${lineIndex}`}>
                  {line}
                  {lineIndex < array.length - 1 && <br />}
                </React.Fragment>
              ));
            })}
          </span>
        );
      }
    });
  };

  return <div className="formatted-message">{parseMessage(content)}</div>;
};

// Enhanced component with comprehensive formatting
const EnhancedMessageFormatter = ({ content }) => {
  const formatContent = () => {
    const lines = content.split('\n');
    const formatted = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      if (!line) {
        formatted.push(<br key={i} />);
        continue;
      }
      
      // Check for numbered lists
      const numberedMatch = line.match(/^(\d+\.)\s*\*\*(.*?)\*\*(.*)/);
      if (numberedMatch) {
        formatted.push(
          <div key={i} className="mb-2 pl-4">
            <span className="font-semibold text-blue-600">{numberedMatch[1]}</span>{' '}
            <strong>{numberedMatch[2]}</strong>
            <span>{numberedMatch[3]}</span>
          </div>
        );
        continue;
      }
      
      // Check for regular bold text
      const boldParts = line.split(/(\*\*[^*]+\*\*)/);
      const formattedLine = boldParts.map((part, partIndex) => {
        if (part.match(/^\*\*.*\*\*$/)) {
          return <strong key={partIndex}>{part.replace(/\*\*/g, '')}</strong>;
        }
        return part;
      });
      
      formatted.push(
        <div key={i} className="mb-1">
          {formattedLine}
        </div>
      );
    }
    
    return formatted;
  };
  
  return <div className="formatted-message space-y-1">{formatContent()}</div>;
};

// Demo component showing all three approaches
const MessageFormattingDemo = ({ content }) => {

  return (
          <div>
            <SafeFormattedMessage content={content} />
          </div>
  );
};

// Updated ChatWindow component with formatted messages
const UpdatedChatWindow = ({ persona, onBack }) => {
    const [messages, setMessages] = useState([]);
    const [newMessage, setNewMessage] = useState('');
    const messagesEndRef = useRef(null);

    // Enhanced message formatter component
    const MessageFormatter = ({ content }) => {
        const formatContent = () => {
            const lines = content.split('\n');
            const formatted = [];
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                
                if (!line) {
                    formatted.push(<br key={i} />);
                    continue;
                }
                
                // Check for numbered lists with bold headers
                const numberedMatch = line.match(/^(\d+\.)\s*\*\*(.*?)\*\*(.*)/);
                if (numberedMatch) {
                    formatted.push(
                        <div key={i} className="mb-2 pl-2">
                            <span className="font-semibold text-blue-600">{numberedMatch[1]}</span>{' '}
                            <strong className="text-gray-800">{numberedMatch[2]}</strong>
                            <span>{numberedMatch[3]}</span>
                        </div>
                    );
                    continue;
                }
                
                // Handle regular bold text
                const boldParts = line.split(/(\*\*[^*]+\*\*)/);
                const formattedLine = boldParts.map((part, partIndex) => {
                    if (part.match(/^\*\*.*\*\*$/)) {
                        return <strong key={partIndex} className="font-semibold">{part.replace(/\*\*/g, '')}</strong>;
                    }
                    return part;
                });
                
                formatted.push(
                    <div key={i} className="mb-1">
                        {formattedLine}
                    </div>
                );
            }
            
            return formatted;
        };
        
        return <div className="formatted-message">{formatContent()}</div>;
    };

    useEffect(() => {
        if (persona) {
            fetch(`http://localhost:8500/api/conversation/${persona.user_id}`)
                .then(res => res.json())
                .then(data => setMessages(data))
                .catch(err => setMessages([]));
        }
    }, [persona]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const handleSendMessage = (e) => {
        e.preventDefault();
        if (newMessage.trim()) {
            const newMsg = {
                role: "student",
                content: newMessage.trim(),
                timestamp: new Date().toISOString(),
            };
            setMessages((prevMessages) => [...prevMessages, newMsg]);
            setNewMessage('');

            setTimeout(() => {
                const tutorResponse = {
                    role: "tutor",
                    content: "This is a simulated tutor response. In a real system, an LLM would generate this.",
                    timestamp: new Date().toISOString(),
                };
                setMessages((prevMessages) => [...prevMessages, tutorResponse]);
            }, 1500);
        }
    };

    return (
        <div className="flex flex-col w-full md:w-2/3 h-[80vh] md:h-[90vh] bg-white rounded-xl shadow-lg">
            {/* Chat Header */}
            <div className="p-4 bg-gradient-to-r from-blue-600 to-blue-800 text-white rounded-t-xl flex items-center justify-between shadow-md">
                <button
                    onClick={onBack}
                    className="text-white hover:text-blue-200 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded-full p-1"
                    aria-label="Back to persona selection"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                    </svg>
                </button>
                <h2 className="text-xl font-semibold">
                    Chat with {persona.persona.charAt(0).toUpperCase() + persona.persona.slice(1)} ({persona.user_id})
                </h2>
                <div className="w-6"></div>
            </div>

            {/* Chat Messages Area */}
            <div className="flex-1 p-4 overflow-y-auto custom-scrollbar bg-gray-50">
                {messages.length === 0 ? (
                    <div className="text-center text-gray-500 mt-10">
                        No messages found for this persona. Start typing to begin a new conversation!
                    </div>
                ) : (
                    messages.map((msg, index) => (
                        <div
                            key={index}
                            className={`flex mb-4 ${msg.role === 'student' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`max-w-[70%] p-3 rounded-xl shadow-sm ${
                                    msg.role === 'student'
                                        ? 'bg-blue-500 text-white rounded-br-none'
                                        : 'bg-white text-gray-800 rounded-bl-none border border-gray-200'
                                }`}
                            >
                                <div className="text-sm">
                                    {/* Use the MessageFormatter component here */}
                                    <MessageFormatter content={msg.content} />
                                </div>
                                <span className="block text-xs opacity-75 mt-2">
                                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                            </div>
                        </div>
                    ))
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Message Input */}
            <form onSubmit={handleSendMessage} className="p-4 border-t border-gray-200 bg-white">
                <div className="flex items-center">
                    <input
                        type="text"
                        value={newMessage}
                        onChange={(e) => setNewMessage(e.target.value)}
                        placeholder="Type your message..."
                        className="flex-1 px-4 py-3 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 text-gray-800"
                    />
                    <button
                        type="submit"
                        className="ml-3 px-6 py-3 bg-green-500 text-white font-semibold rounded-full shadow-md hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        disabled={!newMessage.trim()}
                    >
                        Send
                    </button>
                </div>
            </form>
        </div>
    );
};

export default MessageFormattingDemo;