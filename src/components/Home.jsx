import React from 'react'
import { MatrixEffect } from './MatrixEffect'
import { MdOutlineKeyboardArrowRight } from "react-icons/md"
import { Link } from "react-scroll";

export const Home = () => {
  return (
    <div name="home" className="pt-20 h-screen w-full bg-gradient-to-b from-black to-black text-white scroll-mt-20"> {/* Adjusted padding-top here */}
      <div className="max-w-screen-lg mx-auto flex flex-col items-center justify-center h-full px-4 md:flex-row">
        <div className="flex flex-col justify-center h-full">
          <h2 className='text-4xl sm:text-7xl font-bold text-white'>
            I'm a programmer  
          </h2>
          <p className='text-gray-500 py-4 max-w-md'>
            As a Data Science and Programming enthusiast with a Master’s in Advanced Computer Science with Data Science from the University of Strathclyde, I thrive at the intersection of data analysis, machine learning, and software technology innovation. My journey has been marked by a relentless pursuit of understanding complex patterns within data, leading to impactful insights. From predictive modeling for song popularity to developing real-time facial recognition systems, my projects underscore a commitment to solving real-world problems through technology. Beyond the technical, my aim is to contribute to a future where technology enhances human experiences. I am driven by curiosity and a deep belief in data science's transformative power.                 
          </p>

          <div>
            <Link to="portfolio" smooth duration={500} className='group text-white w-fit px-6 py-3 my-2 flex items-center rounded-md bg-gradient-to-r from-green-800 to-green-500 cursor-pointer'>
              Portfolio
              <span className='group-hover:rotate-90 duration-300 '>
                <MdOutlineKeyboardArrowRight size={25} className="ml-1"/> 
              </span>
            </Link>
          </div>
        </div>
        <div className="w-full md:w-1/2 md:h-1/2 h-full relative">
          <div className="absolute inset-0 bg-black bg-opacity-20 backdrop-filter backdrop-blur-sm">
            <MatrixEffect />
          </div>
          <div className="absolute inset-0 pointer-events-none border border-transparent" 
               style={{ boxShadow: 'inset 0 0 10px 10px rgba(0, 0, 0, 0.5)' }}>
            {/* Invisible overlay to create the fading border effect */}
          </div>
        </div>
      </div>
    </div>
  );
};