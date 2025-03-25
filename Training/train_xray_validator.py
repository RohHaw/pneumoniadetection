from xray_validator import ChestXrayValidator

validator = ChestXrayValidator()
validator.train(
    xray_dirs=['D:\\OneDrive\\OneDrive - University of Leeds\\Uni\\Year 3\\Dissertation\\xray_validator_dataset\\chest_xrays\\normal', 
           'D:\\OneDrive\\OneDrive - University of Leeds\\Uni\\Year 3\\Dissertation\\xray_validator_dataset\\chest_xrays\\pneumonia'],
    non_xray_dir = 'D:\\OneDrive\\OneDrive - University of Leeds\\Uni\\Year 3\\Dissertation\\xray_validator_dataset\\non_xrays',
    epochs=10,
    batch_size=32,
    save_path='Training/xray_validator.pth'
)