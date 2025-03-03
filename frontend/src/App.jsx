import React, { useState } from "react";
import "./App.css";
import { InputComponent } from "./components/inputComponent";
import { MessagesComponent } from "./components/messagesComponent";

function App() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! How can I assist you today?" },
  ]);

  const handleSendMessage = async (newMessage) => {
    // Add the user's message to the conversation
    setMessages((prevMessages) => [
      ...prevMessages,
      { sender: "user", text: newMessage },
    ]);

    try {
      // Send the query to the backend API
      const response = await fetch("http://localhost:8000/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: newMessage }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch response from the server.");
      }

      const data = await response.json();
      const botResponse = data.results[0];

      // Add the bot's response to the conversation
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: botResponse },
      ]);
    } catch (error) {
      console.error("Error:", error);
      // Add an error message to the conversation
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: "Sorry, I couldn't process your request. Please try again." },
      ]);
    }
  };

  return (
    <main className="main-wrapper">
      <div className="body-container">
        <MessagesComponent messages={messages} />
        <InputComponent onSendMessage={handleSendMessage} />
      </div>
    </main>
  );
}

export default App;





