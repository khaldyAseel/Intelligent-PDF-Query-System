import React,{useRef,useEffect} from "react";
import './messagesStyle.css';

export const MessagesComponent = ({ messages, isLoading }) => {
    const messagesEndRef = useRef(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="messages-container">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`message ${message.sender === "user" ? "user-message" : "bot-message"}`}
        >
          <p>{message.text}</p>
        </div>
      ))}
      <div ref={messagesEndRef} />
      {isLoading && <div className="loading-spinner"></div>}
    </div>
  );
};