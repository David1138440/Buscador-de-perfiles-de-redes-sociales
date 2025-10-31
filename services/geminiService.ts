import { GoogleGenAI } from "@google/genai";

// A "real" app would have a more robust way of handling API keys.
// For this example, we'll assume it's set in the environment.
const API_KEY = process.env.API_KEY;

if (!API_KEY) {
    throw new Error("API_KEY environment variable not set.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

/**
 * Converts a File object to a base64 encoded string.
 */
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
        // result is a data URL like "data:image/jpeg;base64,...."
        // We only want the base64 part
        const base64String = (reader.result as string).split(',')[1];
        resolve(base64String);
    };
    reader.onerror = (error) => reject(error);
  });
}

/**
 * Analyzes an image using the Gemini API.
 * @param imageFile The image file to analyze.
 * @returns The text analysis from the Gemini API.
 */
export async function analyzeImageWithGemini(imageFile: File): Promise<string> {
  try {
    const base64Image = await fileToBase64(imageFile);
    
    const imagePart = {
      inlineData: {
        mimeType: imageFile.type,
        data: base64Image,
      },
    };

    const textPart = {
      text: "Describe this image in detail. What is happening, what are the objects, and what is the overall mood?",
    };

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: { parts: [imagePart, textPart] },
    });

    return response.text;
  } catch (error) {
    console.error("Error analyzing image with Gemini:", error);
    // Let's provide a more user-friendly error message
    if (error instanceof Error) {
        if (error.message.includes('API key not valid')) {
            throw new Error('The configured API key is invalid. Please check your setup.');
        }
        throw new Error(`An error occurred while communicating with the API: ${error.message}`);
    }
    throw new Error("An unexpected error occurred during image analysis.");
  }
}
