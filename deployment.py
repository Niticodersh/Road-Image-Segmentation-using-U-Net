import streamlit as st
import torch
import torch.nn as nn
import io
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=256, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels))
        return block

    def forward(self, X):
        contracting_11_out = self.contracting_11(X)

        contracting_12_out = self.contracting_12(contracting_11_out)

        contracting_21_out = self.contracting_21(contracting_12_out)

        contracting_22_out = self.contracting_22(contracting_21_out)

        contracting_31_out = self.contracting_31(contracting_22_out)

        contracting_32_out = self.contracting_32(contracting_31_out)

        middle_out = self.middle(contracting_32_out)

        expansive_11_out = self.expansive_11(middle_out)

        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_31_out), dim=1))

        expansive_21_out = self.expansive_21(expansive_12_out)

        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_21_out), dim=1))

        expansive_31_out = self.expansive_31(expansive_22_out)

        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_11_out), dim=1))

        output_out = self.output(expansive_32_out)
        return output_out


# Function to perform image segmentation
def segment_image(input_image):
    # Load the input image
    image = Image.open(io.BytesIO(input_image)).convert('RGB')
    # image = np.array(image)

    # Perform image segmentation using your model
    # Replace 'segmentation_function' with your actual segmentation code
    model1 = Net(num_classes=12)
    model2 = Net(num_classes=12)
    model3 = Net(num_classes=12)

    segmentation_model_path1 = "Model_1.pth"
    segmentation_model_path2 = "Model_2.pth"
    segmentation_model_path3 = "Model_3.pth"

    if torch.cuda.is_available():
        state_dict1 = torch.load(segmentation_model_path1)
        state_dict1 = {key.replace("module.", ""): value for key, value in state_dict1.items()}
        model1.load_state_dict(state_dict1)

        state_dict2 = torch.load(segmentation_model_path2)
        state_dict2 = {key.replace("module.", ""): value for key, value in state_dict2.items()}
        model2.load_state_dict(state_dict2)

        state_dict3 = torch.load(segmentation_model_path3)
        state_dict3 = {key.replace("module.", ""): value for key, value in state_dict3.items()}
        model3.load_state_dict(state_dict3)
    else:
        state_dict1 = torch.load(segmentation_model_path1, map_location=torch.device('cpu'))
        state_dict1 = {key.replace("module.", ""): value for key, value in state_dict1.items()}
        model1.load_state_dict(state_dict1)

        state_dict2 = torch.load(segmentation_model_path2, map_location=torch.device('cpu'))
        state_dict2 = {key.replace("module.", ""): value for key, value in state_dict2.items()}
        model2.load_state_dict(state_dict2)

        state_dict3 = torch.load(segmentation_model_path3, map_location=torch.device('cpu'))
        state_dict3 = {key.replace("module.", ""): value for key, value in state_dict3.items()}
        model3.load_state_dict(state_dict3)
    h, w = 128, 128
    mean = [0., 0., 0.]
    std = [1., 1., 1.]
    test_transformer = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    image = test_transformer(image).unsqueeze(0)
    #  image = from_numpy(image).permute(2, 0, 1)
    pred1 = model1(image)
    pred2 = model2(image)
    pred3 = model3(image)

    segmented_image1 = pred1[0].permute(1, 2, 0)
    segmented_image1 = torch.argmax(segmented_image1, dim=2).cpu()

    segmented_image2 = pred2[0].permute(1, 2, 0)
    segmented_image2 = torch.argmax(segmented_image2, dim=2).cpu()

    segmented_image3 = pred3[0].permute(1, 2, 0)
    segmented_image3 = torch.argmax(segmented_image3, dim=2).cpu()

    segmented_image4 = (segmented_image1 + segmented_image2 + segmented_image3) / 3
    # segmented_image = Image.fromarray(segmented_image)
    # Create a matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    plt.title("Segmented Images")
    # Display segmented_image1
    axes[0, 0].imshow(segmented_image1, aspect='equal')
    axes[0, 0].set_title('From Model 1')
    axes[0, 0].axis('off')

    # Display segmented_image2
    axes[0, 1].imshow(segmented_image2, aspect='equal')
    axes[0, 1].set_title('From Model 2')
    axes[0, 1].axis('off')

    # Display segmented_image3
    axes[1, 0].imshow(segmented_image3, aspect='equal')
    axes[1, 0].set_title('From Model 3')
    axes[1, 0].axis('off')

    # Display segmented_image4
    axes[1, 1].imshow(segmented_image4, aspect='equal')
    axes[1, 1].set_title('Ensemble Model')
    axes[1, 1].axis('off')

    st.pyplot(fig)
    # return segmented_image


# Streamlit App
st.title('Image Segmentation App')

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Perform segmentation when a button is clicked
    if st.button('Segment Image'):
        with st.spinner('Segmenting...'):
            # Call the segment_image function with the uploaded image
            segment_image(uploaded_image.getvalue())
