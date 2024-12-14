const express = require("express");
const multer = require("multer");
const { v4: uuidv4 } = require("uuid");
const moment = require("moment");
const tf = require("@tensorflow/tfjs-node");
const { Storage } = require("@google-cloud/storage");
const { Firestore } = require("@google-cloud/firestore");
const sharp = require("sharp");
const admin = require("firebase-admin");

const port = 8080;

const app = express();

// Firestore setup
const firestore = new Firestore({
    projectId: "submissionmlgc-christopher",
});

// Initialize Firebase Admin SDK
admin.initializeApp({
    credential: admin.credential.cert(require("./firebase-serviceAccount.json")), // please replace with your firebase admin account
});

const db = admin.firestore();
const collectionName = "predictions"; // Firestore collection

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
    try {
        console.log("Model loaded successfully.");
        return tf.loadGraphModel('https://storage.googleapis.com/submissionmlgc-christopher/submissions-model/model.json');
    }
    catch {
        console.error("Error loading model:", error);
        throw error;
    }
}

// Prediction function
async function predictCancer(image) {
    const model = await loadModel();
    console.log(image);
    let imgTensor = tf.node.decodeImage(image);
    if (imgTensor.shape[2] === 4) {
        imgTensor = imgTensor.slice([0, 0, 0], [-1, -1, 3]);
    }
    imgTensor = imgTensor
        .resizeNearestNeighbor([224, 224])
        .expandDims()
        .toFloat();

    const prediction = model.predict(imgTensor).dataSync(); // Perform prediction

    // Classify based on prediction
    const result = prediction[0] > 0.5 ? "Cancer" : "Non-cancer";
    return result;
}

// Store prediction in Firestore
async function storePrediction(id, result, createdAt, suggestion) {
    const docRef = db.collection(collectionName).doc(id);
    const data = { id, result, suggestion, createdAt };

    await docRef.set(data); // Save data to Firestore
    console.log(`Prediction with id ${id} saved to Firestore.`);
}

// Endpoint: Predict
app.post("/predict", upload.single("image"), async (req, res) => {
    console.log(req.file);
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

        const predictionResult = await predictCancer(req.file.buffer);

        const id = uuidv4();
        const createdAt = moment().toISOString();

        const suggestion =
            predictionResult === "Cancer"
                ? "Segera periksa ke dokter!"
                : "Anda sehat!";
        console.log(id, predictionResult, createdAt, suggestion);
        await storePrediction(id, predictionResult, createdAt, suggestion);

        console.log(predictionResult);

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
