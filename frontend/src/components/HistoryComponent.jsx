import React from "react";
import logo from "/Users/cleopypavlou/Desktop/aka.svg";
import './historyStyle.css'

export const HistoryComponent = ({ history, onHistoryClick }) => {
  return (
    <div className="history-container">
        <img src={logo} alt="Logo" className="logo" />

      <ul>
        {history.map((query, index) => (
          <li key={index} onClick={() => onHistoryClick(query)}>
            {query}
          </li>
        ))}
      </ul>

      <div className="bottom-logo-container">
            <img src="src/assets/img1.png" />
            <img src="src/assets/img6.webp" />
      </div>
      <p className="bottom-text">Our partners dasdf ad a</p>
    </div>
  );
};