import React from 'react'
import HeroImage from '../imports/IMG_0098.jpg';
import { FaGithub, FaLinkedin } from "react-icons/fa";
import { HiOutlineMail } from "react-icons/hi";
import { BsFillPersonLinesFill } from "react-icons/bs";
export const About = () => {
  const socialLinks = [
    {
      id: 1,
      icon: <FaLinkedin size={30} />,
      name: 'LinkedIn',
      href: "https://www.linkedin.com/in/k-adil-rahman-327905182",
    },
    {
      id: 2,
      icon: <FaGithub size={30} />,
      name: 'GitHub',
      href: "https://github.com/kadilrahman",
    },
    {
      id: 3,
      icon: <HiOutlineMail size={30} />,
      name: 'Mail',
      href: "mailto:rehman.adil17@gmail.com",
    },
    {
      id: 4,
      icon: <BsFillPersonLinesFill size={30} />,
      name: 'Resume',
      href: "https://docs.google.com/document/d/1xInWROoyDhZx9kQSWSvVxrqYMtkRpY_c/edit?usp=sharing&ouid=100084612999185821456&rtpof=true&sd=true",
    },
  ];
  
  return (
    
    <div
      name="about"
      className="w-full bg-gradient-to-b from-black to-gray-800 text-white overflow-hidden"
    >
      <div className="max-w-screen-lg p-4 mx-auto flex flex-col justify-center min-h-screen md:min-h-0 md:h-auto sm:pt-32">
        <div className="pb-8 pt-20 md:pt-20">
          <p className="text-4xl font-bold inline border-b-4 border-gray-500">
            About
          </p>
        </div>

        {/* Flex container for image and content */}
        <div className="flex flex-col md:flex-row items-center justify-center">
          
          <div className="text-xl mt-5 md:mt-0 md:ml-8">
          <p className="mt-6">
            Welcome to my professional portfolio. I'm K. Adil Rahman, a Data Science enthusiast with a Master’s in Advanced Computer Science from the University of Strathclyde, Glasgow, UK. My journey is a testament to my dedication to using data and technology to solve real-world problems.
          </p>
          <p className="mt-4">
            My academic path emphasized data science and computer vision, with projects from predictive modeling to real-time facial recognition using CNNs. These experiences honed my technical skills and fostered a spirit of collaboration and innovation.
          </p>
          <p className="mt-4">
            Transitioning from academia to industry, my internship as a React.js Developer at AK Technology was a significant milestone, enhancing my development skills and understanding of team dynamics.
          </p>
          <p className="mt-4">
            My diverse skill set includes SQL, Python, TensorFlow, PyTorch, Keras, React.js, and AWS. This toolkit ensures I am well-versed in the development and deployment of scalable solutions.
          </p>
          <p className="mt-4">
            My commitment to excellence was recognized when I led a project to victory in an Application Development competition, underlining my capability in delivering impactful technology solutions.
          </p>
          <p className="mt-4">
            Driven by the belief that thoughtful technology can enhance our understanding and improve lives, my goal is to explore the depths of data science and machine learning, contributing to boundary-pushing projects.
          </p>
          <p className="mt-4">
            I invite you to explore my portfolio and join me on this exciting journey in data science and technology.
          </p>
          </div>
          {/* Image container */}
          <img src={HeroImage} alt="About" className="rounded-2xl mx-auto w-2/3 md:w-1/4 lg:w-1/4 mt-4 " />
          
          {/* Text content container */}
        </div>
         {/* Social Links for Mobile View */}
         <div className="mt-6 lg:hidden flex justify-center space-x-4">
          {socialLinks.map(({ id, icon, href }) => (
            <a key={id} href={href} target="_blank" rel="noopener noreferrer" className="text-white hover:text-gray-400">
              {icon}
            </a>
          ))}
        </div>
      </div>
    </div>
  );
};