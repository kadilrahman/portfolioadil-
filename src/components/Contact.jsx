import React from 'react'

export const Contact = () => {
    return (
      <div
        name="contact"
        className="w-full h-screen bg-gradient-to-b from-gray-800 to-black p-4 text-white flex items-center justify-center"
      >
        <div className="flex flex-col justify-center items-center w-full max-w-4xl p-8 rounded-lg bg-gray-800 shadow-lg">
          <div className="pb-8 text-center">
            <p className="text-4xl font-bold inline border-b-4 border-cyan-500">
              Contact
            </p>
            <p className="py-6">Ready to chat? Drop a message below and I'll get back to you ASAP.</p>
          </div>
  
          <form
            action="https://getform.io/f/raeggnla"
            method="POST"
            className="flex flex-col w-full md:w-3/4"
          >
            <input
              type="text"
              name="name"
              placeholder="Your Name"
              className="p-3 bg-transparent border-2 rounded-md text-white focus:outline-none focus:border-cyan-500 transition duration-200 ease-in-out"
            />
            <input
              type="email"
              name="email"
              placeholder="Your Email"
              className="my-4 p-3 bg-transparent border-2 rounded-md text-white focus:outline-none focus:border-cyan-500 transition duration-200 ease-in-out"
            />
            <textarea
              name="message"
              placeholder="Your Message"
              rows="6"
              className="p-3 bg-transparent border-2 rounded-md text-white focus:outline-none focus:border-cyan-500 transition duration-200 ease-in-out"
            ></textarea>
  
            <button className="text-white bg-cyan-500 hover:bg-cyan-600 px-6 py-3 my-8 mx-auto flex items-center rounded-md transition duration-300 ease-in-out">
              Send Message
            </button>
          </form>
        </div>
      </div>
    );
  };