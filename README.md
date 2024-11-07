# batch-lector

# Markdown Batch Processor

A robust Python script for processing large markdown files using Anthropic's Claude API Batching feature. Specifically designed for fixing markdown formatting and structural issues in OCR-scanned documents while preserving the original content.

## Features

- **Batch Processing**: Process large markdown files in chunks of up to 10,000 requests per batch
- **Cost-Efficient**: Utilizes Anthropic's batch processing API for 50% cost reduction
- **Content Preservation**: Fixes only structural and formatting issues while maintaining original content
- **Error Handling**: Comprehensive error handling with detailed logging
- **Progress Monitoring**: Real-time progress tracking and notifications via Gotify
- **Resume Capability**: Ability to resume processing from a specific batch ID
- **Dry Run Mode**: Test connectivity without processing actual content

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/markdown-batch-processor.git
cd markdown-batch-processor
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install anthropic requests
```

4. Set up environment variables:
```bash
export ANTHROPIC_API_KEY="your-api-key"
export GOTIFY_TOKEN="your-gotify-token"  # Optional
```

## Usage

### Basic Usage

Process a markdown file:
```bash
python processor.py --input path/to/your/file.md
```

### Command Line Arguments

```
--input          Required: Input markdown file path
--dry-run        Optional: Test connections without processing
--gotify-url     Optional: Gotify server URL (default: https://push.example.de)
--gotify-token   Optional: Gotify notification token
--api-key        Optional: Anthropic API key (can also use ANTHROPIC_API_KEY env variable)
--resume-batch   Optional: Resume processing from an existing batch ID
```

### Examples

1. Run with all parameters:
```bash
python processor.py --input document.md --gotify-url https://your-gotify-server.com --gotify-token your-token --api-key your-api-key
```

2. Test connections:
```bash
python processor.py --input document.md --dry-run
```

3. Resume from a batch:
```bash
python processor.py --input document.md --resume-batch msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d
```

## Notifications

The script uses Gotify for notifications with different priority levels:
- Priority 5: Information and status updates
- Priority 7: Batch completion and warnings
- Priority 8-9: Processing errors
- Priority 10: Fatal errors

## Error Handling

- Comprehensive error catching and reporting
- Partial results saving on failure
- Detailed error messages with stack traces
- Notification system for real-time error alerts

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and follow the existing coding style.