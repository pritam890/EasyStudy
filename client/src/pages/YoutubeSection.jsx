import { useEffect, useState } from 'react';
import { Youtube } from 'lucide-react';

function YoutubeSection() {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchYouTubeVideos = async () => {
    try {
      const res = await fetch('http://localhost:5000/youtube');
      const data = await res.json();
      if (data.success) {
        setVideos(data.videos);
      } else {
        setError('Failed to fetch YouTube videos.');
      }
    } catch (err) {
      setError('Error fetching YouTube videos.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchYouTubeVideos();
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <div className="mb-8 flex items-center gap-2 text-purple-700">
        <Youtube className="w-8 h-8" />
        <h2 className="text-3xl font-bold">🎥 Related YouTube Videos</h2>
      </div>

      {loading && <p className="text-gray-600 text-lg">Loading videos...</p>}

      {error && (
        <p className="text-red-500 text-lg mt-4">
          {error}
        </p>
      )}

      {!loading && !error && (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {videos.map((video, index) => (
            <div
              key={index}
              className="border border-gray-200 rounded-xl shadow hover:shadow-lg transition p-3 bg-white"
            >
              <a
                href={`https://www.youtube.com/watch?v=${video.video_id}`}
                target="_blank"
                rel="noopener noreferrer"
              >
                <img
                  src={video.thumbnail}
                  alt={video.title}
                  className="rounded-lg w-full h-48 object-cover mb-3"
                />
                <p className="text-gray-800 font-medium text-md hover:text-purple-600 transition">
                  {video.title}
                </p>
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default YoutubeSection;
