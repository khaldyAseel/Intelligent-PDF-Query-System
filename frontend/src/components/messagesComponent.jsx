import React from "react";

export const MessagesComponent = ({ messages }) => {
  return (
    <div className="messages-container">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`message ${message.sender === "user" ? "user-message" : "bot-message"}`}
        >
          {message.text}
        </div>
      ))}
    </div>
  );
};