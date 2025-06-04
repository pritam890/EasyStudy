import React, { useEffect, useState } from 'react';
import axios from 'axios';

const YouTube = () => {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchYouTubeVideos = async () => {
      try {
        const response = await axios.get('http://localhost:5000/youtube');
        setVideos(response.data.videos || []);
      } catch (error) {
        console.error('Error fetching YouTube videos:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchYouTubeVideos();
  }, []);

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
        Trending YouTube Videos
      </h2>

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 animate-pulse">
          {Array.from({ length: 6 }).map((_, index) => (
            <div key={index} className="bg-gray-200 rounded-lg h-64"></div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {videos.map((video, index) => (
            <div
              key={index}
              className="rounded-xl shadow-lg bg-white overflow-hidden transition-transform transform hover:-translate-y-1 hover:shadow-2xl"
            >
              <a
                href={`https://www.youtube.com/watch?v=${video.video_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="block"
              >
                <img
                  src={video.thumbnail}
                  alt={video.title}
                  onError={(e) => {
                    e.target.src = '/fallback-thumbnail.jpg';
                  }}
                  className="w-full h-48 object-cover transition-opacity duration-300 hover:opacity-90"
                />
                <div className="p-3">
                  <h3 className="text-lg font-semibold text-gray-800 truncate">{video.title}</h3>
                  <p className="text-sm text-gray-500 mt-1 truncate">Watch on YouTube</p>
                </div>
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default YouTube;
