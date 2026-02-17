import axios from "axios";

const api = axios.create({
  baseURL: "https://topsis-102317200-k28h.vercel.app/api",
});

export default api;
