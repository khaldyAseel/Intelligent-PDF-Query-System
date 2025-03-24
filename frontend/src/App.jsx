import React, { useState } from "react";
import "./App.css";
import { InputComponent } from "./components/inputComponent";
import { MessagesComponent } from "./components/messagesComponent";
import { HistoryComponent } from "./components/historyComponent";

function App() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! How can I assist you today?" },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState([]); // State for history

  const handleSendMessage = async (newMessage) => {
    setIsLoading(true); // Start loading
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

      // Add the query to the history
      setHistory((prevHistory) => [...prevHistory, newMessage]);
    } catch (error) {
      console.error("Error:", error);
      // Add an error message to the conversation
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: "Sorry, I couldn't process your request. Please try again." },
      ]);
    } finally {
      setIsLoading(false); // Stop loading
    }
  };

  const handleHistoryClick = (query) => {
    // When a history item is clicked, set the messages to show the query and its response
    setMessages([
      { sender: "bot", text: "Hello! How can I assist you today?" },
      { sender: "user", text: query },
      { sender: "bot", text: `${query}` }, // Replace this with actual response logic if needed
    ]);
  };

  return (
    <main className="main-wrapper">
      <div className="history-sidebar">
        <HistoryComponent history={history} onHistoryClick={handleHistoryClick} />
      </div>
      <div className="body-container">
        <MessagesComponent isLoading={isLoading} messages={messages} />
        <InputComponent onSendMessage={handleSendMessage} />
      </div>
    </main>
  );
}

export default App;