const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const cocoSsd = require("@tensorflow-models/coco-ssd");
const cors = require("cors");
const fs = require("fs");

const app = express();
app.use(cors());
app.use(express.json());

// Set up Multer for image uploads
const upload = multer({ dest: "uploads/" });

// Load the COCO-SSD model
let model;
(async () => {
  model = await cocoSsd.load();
  console.log("COCO-SSD model loaded.");
})();

app.post("/upload-image", upload.single("image"), async (req, res) => {
  if (!model) {
    return res
      .status(503)
      .json({ error: "Model not loaded yet. Please try again later." });
  }

  try {
    const imagePath = req.file.path;
    const imageBuffer = fs.readFileSync(imagePath);
    const decodedImage = tf.node.decodeImage(imageBuffer);

    // Perform object detection
    const predictions = await model.detect(decodedImage);

    // Prepare objects and bounding boxes
    const detectedObjects = predictions.map((prediction) => ({
      class: prediction.class,
      bbox: prediction.bbox, 
    }));

    // Send detected objects and bounding boxes as response
    res.json({
      objects: detectedObjects.map((obj) => obj.class),
      boxes: detectedObjects,
    });

    // Clean up
    fs.unlinkSync(imagePath);
    decodedImage.dispose();
  } catch (error) {
    console.error("Error during image processing:", error);
    res.status(500).json({ error: "Failed to process image" });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
