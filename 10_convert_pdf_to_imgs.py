import logging
from helpers import ProcessingError, convert_pdf_to_images, SRC_DIR, IMG_DIR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Create directories
        for path in [SRC_DIR, IMG_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        # Process PDFs
        src_file_paths = list(SRC_DIR.glob("*.pdf"))
        if not src_file_paths:
            logger.warning("No PDF files found in source directory")
            return

        for src_file_path in src_file_paths:

            try:
                logger.info(f"Processing {src_file_path}...")
                convert_pdf_to_images(src_file_path, IMG_DIR)
            except ProcessingError as e:
                logger.error(f"Failed to process {src_file_path}: {str(e)}")
                continue


    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
