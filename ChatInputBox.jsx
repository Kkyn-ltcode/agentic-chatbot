import React from 'react';
import './style.css';

const ChatInputBox = () => {
  return (
    <div className='chat-input-container'>
      <div className='top-layer'>
        <input
          type='text'
          placeholder='Reply to Claude...'
          className='chat-input-field'
        />
      </div>
      <div className='bottom-layer'>
        <div className='chat-input-left'>
          <button className='icon-button'>+</button>
          <button className='icon-button separator-icon'>⇄</button>
        </div>
        <div className='chat-input-right'>
          <div className='model-select'>
            <span>Claude Sonnet 4</span>
            <span className='dropdown-arrow'>↓</span>
          </div>
          <button className='send-button'>↑</button>
        </div>
      </div>
    </div>
  );
};

export default ChatInputBox;
