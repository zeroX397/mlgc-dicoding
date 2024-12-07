const express = require("express");
const multer = require("multer");
const { v4: uuidv4 } = require("uuid");
const moment = require("moment");
const tf = require("@tensorflow/tfjs");
const { Storage } = require("@google-cloud/storage");
const { Firestore } = require("@google-cloud/firestore");
const sharp = require("sharp");

const port = 3000;

const app = express();
// const port = process.env.PORT || 8080;

// Google Cloud Storage setup
const storage = new Storage();
const bucketName = "submissionmlgc-christopher";
const modelPath = "submissions-model/model.json";
let model;

// Firestore setup
const firestore = new Firestore();
const collectionName = "prediction_histories"; // Firestore collection

// Configure multer for file upload
const upload = multer({ 
    limits: { fileSize: 1000000 }, // Max size: 1MB
    fileFilter(req, file, cb) {
        if (!file.mimetype.startsWith("image/")) {
            return cb(new Error("File must be an image"));
        }
        cb(null, true);
    },
});

// Load model from Cloud Storage
async function loadModel() {
    const bucket = storage.bucket(bucketName);
    const file = bucket.file(modelPath);
  
    const [exists] = await file.exists();
    if (!exists) {
      throw new Error(`File ${modelPath} does not exist in bucket ${bucketName}`);
    }
  
    const [fileData] = await file.download();
    model = await tf.loadLayersModel(tf.io.fromMemory(fileData.toString()));
    console.log("Model loaded successfully");
  }
  

// Prediction function
async function predictCancer(imageBuffer) {
    const imageTensor = tf.node.decodeImage(imageBuffer, 3); // Decode RGB image
    const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]); // Resize to 224x224
    const input = resizedImage.expandDims(0).div(255.0); // Normalize
    const prediction = model.predict(input).dataSync(); // Perform prediction

    // Classify based on prediction
    const result = prediction[0] > 0.5 ? "Cancer" : "Non-cancer";
    return result;
}

// Store prediction in Firestore
async function storePrediction(id, result, createdAt, suggestion) {
    const history = {
        id,
        result,
        createdAt,
        suggestion,
    };
    await firestore.collection(collectionName).doc(id).set({ history });
}

// Endpoint: Predict
app.post("/predict", upload.single("image"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                status: "fail",
                message: "File not provided",
            });
        }

        const imageBuffer = await sharp(req.file.buffer)
            .resize(224, 224)
            .toBuffer();

        const predictionResult = await predictCancer(imageBuffer);

        const id = uuidv4();
        const createdAt = moment().toISOString();

        const suggestion =
            predictionResult === "Cancer"
                ? "Segera periksa ke dokter!"
                : "Anda sehat!";

        await storePrediction(id, predictionResult, createdAt, suggestion);

        res.status(200).json({
            status: "success",
            message: "Model is predicted successfully",
            data: {
                id,
                result: predictionResult,
                suggestion,
                createdAt,
            },
        });
    } catch (error) {
        console.error(error);
        res.status(400).json({
            status: "fail",
            message: "Terjadi kesalahan dalam melakukan prediksi",
        });
    }
});

// Endpoint: Prediction Histories
app.get("/predict/histories", async (req, res) => {
    try {
        const snapshot = await firestore.collection(collectionName).get();
        const histories = [];

        snapshot.forEach((doc) => {
            const data = doc.data();
            histories.push({
                id: doc.id,
                ...data,
            });
        });

        res.status(200).json({
            status: "success",
            data: histories,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            status: "fail",
            message: "Gagal mengambil data histories",
        });
    }
});

// Handle errors from multer
app.use((err, req, res, next) => {
    if (err instanceof multer.MulterError) {
        return res.status(413).json({
            status: "fail",
            message: `Payload content length greater than maximum allowed: 1000000`,
        });
    }
    if (err.message === "File must be an image") {
        return res.status(400).json({
            status: "fail",
            message: "File must be an image",
        });
    }
    next(err);
});

// Start server and load model
app.listen(port, async () => {
    await loadModel(); // Ensure model is loaded before accepting requests
    console.log(`Server running at http://localhost:${port}`);
});
