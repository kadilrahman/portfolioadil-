import React from 'react'

import python from "../imports/Python-logo-notext.svg.png";
import sql from "../imports/sql.png";
import tensorflow from "../imports/Tensorflow_logo.svg.png";
import pytorch from "../imports/Pytorch .png";
import pandas from "../imports/pandas.png";
import numpy from "../imports/numpy.png";
import matplot from "../imports/mathplot.png";
import sk from "../imports/sk.png"
import javascript from "../imports/JavaScript-logo.png";
import react from "../imports/react.png";
import mongodb from "../imports/mongo.png";
import json from "../imports/json.png";
import hdfs from "../imports/hdfs.png";
import spark from "../imports/spark2.png";
import excel from "../imports/excel.png";
import xml from "../imports/xml.png";

export const Experience = () => {
  const techs = [
    {
      id: 1,
      src: python,
      title: "Pythin",
      style: " shadow-blue-500",
    },
    
    {
      id: 2,
      src: sql,
      title: "SQL",
      style: "shadow-red-500",
    },

    {
      id: 3,
      src: tensorflow,
      title: "TensorFlow",
      style: "shadow-orange-500",
    },
    
    {
      id: 4,
      src: pytorch,
      title: "Pytorch",
      style: "shadow-red-500",
    },

    {
      id: 5,
      src: pandas,
      title: "Pandas",
      style: " shadow-blue-500",
    },
    {
      id: 6,
      src: numpy,
      title: "NumPy",
      style: " shadow-blue-500",
    },
    {
      id: 6,
      src: matplot,
      title: "Matplotlib",
      style: " shadow-gray-500",
    },
    {
      id: 6,
      src: sk,
      title: "Scikit-learn",
      style: " shadow-orange-400",
    },
    {
      id: 7,
      src: javascript,
      title: "Javascript",
      style: "shadow-yellow-400",
    },
    {
      id: 8,
      src: react,
      title: "React",
      style: "shadow-sky-400",
    },
    {
      id: 9,
      src: mongodb,
      title: "MongoDB",
      style: "shadow-green-400",
    },
    {
      id: 10,
      src: json,
      title: "JSON",
      style: "shadow-purple-400",
    },
    {
      id: 10,
      src: xml,
      title: "XML",
      style: "shadow-purple-400",
    },
    {
      id: 11,
      src: hdfs,
      title: "HDFS",
      style: "shadow-yellow-300",
    },
    {
      id: 12,
      src: spark,
      title: "Apache Spark",
      style: "shadow-orange-400",
    },
    {
      id: 13,
      src: excel,
      title: "Microsoft Excel",
      style: "shadow-orange-400",
    },
  ];

  return (
    <div name="experience" className="bg-gradient-to-b from-black to-gray-800 w-full overflow-hidden">
      <div className="max-w-screen-lg mx-auto p-4 text-white">
        <div>
          <p className="text-4xl font-bold inline border-b-4 border-gray-500 text-left">
            Skills
          </p>
          <p className="py-6">These are the technologies I've worked with:</p>
        </div>

        {/* Scrollable Container */}
        <div className="overflow-auto" style={{ maxHeight: 'calc(100vh - 10rem)' }}> {/* Adjust the maxHeight value as needed */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-8 text-center py-8">
            {techs.map(({ id, src, title, style }) => (
              <div key={id} className={`shadow-md hover:scale-105 duration-500 py-2 rounded-lg ${style} flex flex-col items-center`}>
                <img src={src} alt={title} className="w-20 mx-auto" />
                <p className="mt-4">{title}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};