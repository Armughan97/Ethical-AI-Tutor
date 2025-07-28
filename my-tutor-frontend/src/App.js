import React, { useState, useEffect, useRef } from 'react';
import MessageFormatter from './MessageFormatter';

// Main App component
const App = () => {
    // State to hold the currently selected persona object
    const [selectedPersona, setSelectedPersona] = useState(null);

    // Function to handle persona selection from the dropdown
    const handlePersonaSelect = (persona) => {
        setSelectedPersona(persona);
    };

    // Conditional rendering: show persona selector or chat window
    return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4 font-sans">
            <div className="bg-white shadow-lg rounded-xl w-full max-w-4xl overflow-hidden flex flex-col md:flex-row">
                {selectedPersona ? (
                    // If a persona is selected, display the chat window
                    <ChatWindow
                        persona={selectedPersona}
                        onBack={() => setSelectedPersona(null)} // Allow going back to persona selection
                    />
                ) : (
                    // Otherwise, display the persona selection component
                    <PersonaSelector onSelectPersona={handlePersonaSelect} />
                )}
            </div>
        </div>
    );
};

// Component for selecting a persona
const PersonaSelector = ({ onSelectPersona }) => {
    // Hardcoded personas based on your simulator.py
    const PERSONAS = [
        {
            user_id: "lazy_student_001",
            persona: "lazy",
            description: "Lazy student who wants quick answers with minimal effort",
            behavior: "Stops after 1 turn, asks for direct solutions"
        },
        {
            user_id: "curious_learner_002",
            persona: "curious",
            description: "Curious learner who asks follow-up questions",
            behavior: "Asks 'Why?' and follow-up questions, wants deeper understanding"
        },
        {
            user_id: "persistent_worker_003",
            persona: "persistent",
            description: "Persistent student who rephrases questions when stuck",
            behavior: "Rephrases original question and retries up to 5 times"
        },
        {
            user_id: "strategic_manipulator_004",
            persona: "strategic",
            description: "Strategic student who attempts to bypass restrictions using authority",
            behavior: "Attempts to bypass restrictions using authority"
        },
    ];

    // State to hold the currently selected persona from the dropdown
    const [currentSelection, setCurrentSelection] = useState('');

    // Handle change in the dropdown
    const handleChange = (e) => {
        setCurrentSelection(e.target.value);
    };

    // Handle starting the chat
    const handleStartChat = () => {
        const selected = PERSONAS.find(p => p.user_id === currentSelection);
        if (selected) {
            onSelectPersona(selected);
        }
    };

    // Set initial selection to the first persona if available
    useEffect(() => {
        if (PERSONAS.length > 0 && !currentSelection) {
            setCurrentSelection(PERSONAS[0].user_id);
        }
    }, [PERSONAS, currentSelection]);

    return (
        <div className="p-8 w-full md:w-1/2 mx-auto flex flex-col items-center justify-center h-full">
            <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Select Your Persona</h2>
            <p className="text-gray-600 mb-8 text-center max-w-md">
                Choose a student persona to load their simulated chat history with the AI Tutor.
            </p>
            <div className="w-full max-w-xs mb-6">
                <label htmlFor="persona-select" className="sr-only">Choose a persona</label>
                <select
                    id="persona-select"
                    className="block w-full px-4 py-3 rounded-lg border border-gray-300 bg-white text-gray-900 focus:ring-blue-500 focus:border-blue-500 shadow-sm transition-all duration-200"
                    value={currentSelection}
                    onChange={handleChange}
                >
                    {PERSONAS.map((persona) => (
                        <option key={persona.user_id} value={persona.user_id}>
                            {persona.persona.charAt(0).toUpperCase() + persona.persona.slice(1)} ({persona.user_id})
                        </option>
                    ))}
                </select>
            </div>
            <button
                onClick={handleStartChat}
                disabled={!currentSelection}
                className="px-8 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                Start Chat
            </button>
            {currentSelection && (
                <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800 max-w-md text-center">
                    <h3 className="font-semibold mb-1">Selected Persona: {PERSONAS.find(p => p.user_id === currentSelection)?.persona.charAt(0).toUpperCase() + PERSONAS.find(p => p.user_id === currentSelection)?.persona.slice(1)}</h3>
                    <p>{PERSONAS.find(p => p.user_id === currentSelection)?.description}</p>
                </div>
            )}
        </div>
    );
};

// Component for displaying the chat window
const ChatWindow = ({ persona, onBack }) => {
    // State to hold the chat messages
    const [messages, setMessages] = useState([]);
    // State for the new message input field
    const [newMessage, setNewMessage] = useState('');
    // Ref for scrolling to the bottom of the chat
    const messagesEndRef = useRef(null);

    // Effect to load messages when persona changes
    useEffect(() => {
    if (persona) {
        fetch(`http://localhost:8500/api/conversation/${persona.user_id}`)
            .then(res => res.json())
            .then(data => setMessages(data))
            .catch(err => setMessages([]));
    }
}, [persona]);

    // Effect to scroll to the bottom of the chat window
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Handle sending a new message (simulated)
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

            // Simulate tutor response after a short delay
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
                <div className="w-6"></div> {/* Placeholder for alignment */}
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
                                        : 'bg-gray-200 text-gray-800 rounded-bl-none'
                                }`}
                            >
                                <div className="text-sm">
                                    <MessageFormatter content={msg.content} />
                                </div>
                                <span className="block text-xs opacity-75 mt-1">
                                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                            </div>
                        </div>
                    ))
                )}
                <div ref={messagesEndRef} /> {/* Scroll target */}
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

export default App;
