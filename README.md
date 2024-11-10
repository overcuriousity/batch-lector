# OCR Markdown Processing Tool

A Python tool for processing and fixing OCR-generated markdown files from German texts using Anthropic's Batch API. This tool specifically handles structural and formatting issues while preserving the original German content.

## Features

- Batch processing of large markdown files using Anthropic's cost-effective Batch API
- Preserves original German content while fixing:
  - Markdown structural issues
  - Hyphenation errors
  - Formatting inconsistencies
  - Line breaks and paragraph structures
- Parallel processing of multiple files
- Progress monitoring and notifications via Gotify
- Resumable processing from existing batch IDs
- Support for processing individual files or entire directories

## Prerequisites

- Python 3.8+
- Anthropic API key
- Gotify server for notifications
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your credentials:
```env
ANTHROPIC_API_KEY=your_api_key
GOTIFY_URL=your_gotify_url
GOTIFY_TOKEN=your_gotify_token
SYSTEM_PROMPT=your_custom_prompt  # Optional
```

## Usage

### Basic Usage

Process a single file or directory:
```bash
python main.py --input path/to/your/file_or_directory
```

### Additional Options

- Test connections without processing:
```bash
python main.py --input path/to/file --dry-run
```

- Resume from a previous batch:
```bash
python main.py --input output/path --resume-batch batch_id
```

- Override environment variables:
```bash
python main.py --input path/to/file --gotify-url URL --gotify-token TOKEN --api-key KEY
```

## Code Structure Analysis

### Strengths
- Good separation of concerns with distinct classes and functions
- Comprehensive error handling and notifications
- Flexible file handling supporting both single files and directories
- Efficient batch processing with proper chunking

### Areas for Improvement

1. **Error Handling**:
   - Consider implementing retries for API calls
   - Add more granular error types and handling
   - Implement rate limiting handling

2. **Code Duplication**:
   - Consolidate notification logic into reusable functions
   - Create a unified status tracking system
   - Merge similar batch processing functions

3. **Configuration**:
   - Move hardcoded values to configuration
   - Add support for different OCR/markdown processing strategies
   - Implement configurable batch sizes and timing

4. **Performance**:
   - Optimize chunk size determination
   - Implement smarter batch grouping
   - Add caching for partially processed files

## Environment Variables

| Variable | Description | Required |
|----------|-------------|-----------|
| ANTHROPIC_API_KEY | Your Anthropic API key | Yes |
| GOTIFY_URL | Gotify server URL | Yes |
| GOTIFY_TOKEN | Gotify notification token | Yes |
| SYSTEM_PROMPT | Custom system prompt for processing | No |

## System Prompt

The default system prompt is designed to:
- Fix structural markdown issues
- Correct hyphenation errors
- Preserve original German content
- Maintain original line breaks and paragraph structure
- Remove interfering footnotes
- Preserve image references

You can customize the prompt by setting the `SYSTEM_PROMPT` environment variable.

## Batch Processing Details

- Maximum batch size: 10,000 requests
- Processing time: Up to 24 hours
- Cost: 50% cheaper than standard API calls
- Models supported: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku

## Error Handling and Recovery

The tool implements several recovery mechanisms:
1. Automatic batch status monitoring
2. Progress saving and restoration
3. Partial results recovery
4. Batch processing resumption

## Notifications

Notifications are sent via Gotify for:
- Processing start/completion
- Batch status updates
- Errors and warnings
- Hourly status updates
- Final processing summary

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]