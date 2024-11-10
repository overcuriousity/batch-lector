import anthropic
import sys
import time
import os
import requests
import argparse
import traceback
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

load_dotenv()

@dataclass
class BatchStatus:
    batch_id: str
    chunks: List[Dict[str, str]]
    input_file: str  # Track which file this batch belongs to
    status: str = "in_progress"
    results: Dict[str, str] = field(default_factory=dict)

@dataclass
class ProcessingStatus:
    """Track processing status across multiple files and their batches"""
    active_batches: Dict[str, List[BatchStatus]] = field(default_factory=dict)  # file -> batches
    completed_files: Set[str] = field(default_factory=set)
    failed_files: Dict[str, str] = field(default_factory=dict)  # file -> error message


class GotifyNotifier:
    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = (base_url or os.getenv('GOTIFY_URL', 'https://gotify.example.com')).rstrip('/')
        self.token = token or os.getenv('GOTIFY_TOKEN')
        
        if not self.token:
            raise ValueError("No Gotify token provided. Set GOTIFY_TOKEN in .env file or pass via --gotify-token")
        
    def send_notification(self, title: str, message: str, priority: int = 5):
        """Send notification via Gotify"""
        url = self.base_url
        try:
            response = requests.post(
                url,
                params={'token': self.token},
                data={
                    'title': title,
                    'message': message,
                    'priority': priority
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to send notification: {str(e)}")
            return False

def test_connections(client: anthropic.Anthropic, notifier: GotifyNotifier):
    """Test connections to Anthropic API and Gotify"""
    success = True
    
    # Test Anthropic connection
    try:
        # List batches as a simple API test
        client.beta.messages.batches.list(limit=1)
        print("✓ Successfully connected to Anthropic API")
    except Exception as e:
        print(f"✗ Failed to connect to Anthropic API: {str(e)}")
        success = False
    
    # Test Gotify connection
    if notifier.send_notification(
        "Connection Test",
        "Testing connection to Gotify server",
        priority=5
    ):
        print("✓ Successfully connected to Gotify server")
    else:
        print("✗ Failed to connect to Gotify server")
        success = False
    
    return success

def is_text_file(file_path: Path) -> bool:
    """
    Check if a file is text-based by attempting to read it as text.
    Also checks for common text file extensions as a fast path.
    """
    # Common text file extensions (fast path)
    text_extensions = {
        '.txt', '.md', '.json', '.csv', '.tex', '.rst', 
        '.asc', '.text', '.rtf', '.adoc', '.asciidoc'
    }
    
    if file_path.suffix.lower() in text_extensions:
        return True
        
    # Try reading the file as text
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read first 512 bytes to check if it's text
            sample = f.read(512)
            # Check if sample contains mostly printable characters
            printable_ratio = sum(c.isprintable() or c.isspace() for c in sample) / len(sample)
            return printable_ratio > 0.9  # If 90% of characters are printable, consider it text
    except (UnicodeDecodeError, IOError):
        return False

def get_text_files(input_path: str) -> List[str]:
    """Get all text files from input path (file or directory)"""
    input_path = Path(input_path)
    
    if input_path.is_file():
        return [str(input_path)] if is_text_file(input_path) else []
    elif input_path.is_dir():
        text_files = []
        # Walk through directory recursively
        for path in input_path.rglob('*'):
            if path.is_file() and is_text_file(path):
                text_files.append(str(path))
        return text_files
    return []

def get_batch_status_summary(processing_status: ProcessingStatus, client: anthropic.Anthropic) -> Dict:
    """Get a comprehensive summary of all batch processing status"""
    summary = {
        "files": {
            "total": len(processing_status.active_batches) + len(processing_status.completed_files) + len(processing_status.failed_files),
            "in_progress": len(processing_status.active_batches),
            "completed": len(processing_status.completed_files),
            "failed": len(processing_status.failed_files)
        },
        "batches": {
            "total": sum(len(batches) for batches in processing_status.active_batches.values()),
            "in_progress": 0,
            "completed": 0,
            "error": 0
        },
        "requests": {
            "total": 0,
            "processing": 0,
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0
        },
        "batch_details": {}  # Store details for each batch
    }
    
    # Collect status for all active batches
    for filepath, batches in processing_status.active_batches.items():
        file_batches = []
        for batch in batches:
            try:
                status = client.beta.messages.batches.retrieve(batch.batch_id)
                if status.processing_status == "ended":
                    summary["batches"]["completed"] += 1
                else:
                    summary["batches"]["in_progress"] += 1
                
                # Update request counts
                for count_type, count in status.request_counts.items():
                    summary["requests"][count_type] += count
                
                # Store batch details
                file_batches.append({
                    "id": batch.batch_id,
                    "status": status.processing_status,
                    "requests": status.request_counts
                })
            except Exception as e:
                summary["batches"]["error"] += 1
                file_batches.append({
                    "id": batch.batch_id,
                    "status": "error",
                    "error": str(e)
                })
        
        summary["batch_details"][filepath] = file_batches
    
    return summary

def format_status_message(summary: Dict) -> str:
    """Format status summary into a readable message"""
    message = [
        "Current Processing Status:",
        "\nFiles:",
        f"- Total: {summary['files']['total']}",
        f"- In Progress: {summary['files']['in_progress']}",
        f"- Completed: {summary['files']['completed']}",
        f"- Failed: {summary['files']['failed']}",
        "\nBatches:",
        f"- Total: {summary['batches']['total']}",
        f"- In Progress: {summary['batches']['in_progress']}",
        f"- Completed: {summary['batches']['completed']}",
        f"- Error: {summary['batches']['error']}",
        "\nRequests:",
        f"- Total: {summary['requests']['total']}",
        f"- Processing: {summary['requests']['processing']}",
        f"- Succeeded: {summary['requests']['succeeded']}",
        f"- Failed: {summary['requests']['errored']}",
        f"- Canceled/Expired: {summary['requests']['canceled'] + summary['requests']['expired']}",
        "\nActive Batch Details:"
    ]
    
    for filepath, batches in summary["batch_details"].items():
        message.append(f"\nFile: {Path(filepath).name}")
        for batch in batches:
            status_str = batch['status'].upper()
            if batch['status'] == "error":
                message.append(f"- Batch {batch['id']}: {status_str} ({batch['error']})")
            else:
                req_counts = batch['requests']
                message.append(
                    f"- Batch {batch['id']}: {status_str} "
                    f"({req_counts['succeeded']}/{sum(req_counts.values())} requests completed)"
                )
    
    return "\n".join(message)
    
def chunk_markdown(content: str, chunk_size: int = 50) -> List[Dict[str, str]]:
    """Split markdown content into chunks while preserving paragraph integrity."""
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        current_chunk.append(line)
        current_size += 1
        
        if current_size >= chunk_size or (line.strip() == '' and current_size > 10):
            chunks.append({
                'content': '\n'.join(current_chunk),
                'start_line': len(chunks) * chunk_size + 1
            })
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'start_line': len(chunks) * chunk_size + 1
        })
    
    return chunks

def create_batch_requests(chunks: List[Dict[str, str]], model: str = "claude-3-5-haiku-20241022") -> List[Request]:
    """Create batch requests for markdown processing."""
    # Get system prompt from environment variable, with a fallback default
    system_prompt = os.getenv('SYSTEM_PROMPT', """You are an expert at fixing markdown formatting while preserving the original content. 
    Fix only structural markdown issues, hyphenation errors, and formatting. DO NOT change or translate the German text content.
    Fix all original line breaks and paragraph structures. Return only the fixed markdown without explanations or annotations. Remove any Footnotes which might interfere with the text. Don´t change any image references.""")
    
    return [
        Request(
            custom_id=f"chunk_{i}_{chunk['start_line']}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": chunk['content']}
                ]
            )
        )
        for i, chunk in enumerate(chunks)
    ]

def submit_batches(chunks: List[Dict[str, str]], client: anthropic.Anthropic, input_file: str, notifier: GotifyNotifier, max_batch_size: int = 10000) -> List[BatchStatus]:
    """Submit all chunks in multiple batches if needed."""
    batches = []
    
    # Split chunks into batches of max_batch_size
    for i in range(0, len(chunks), max_batch_size):
        batch_chunks = chunks[i:i + max_batch_size]
        batch_requests = create_batch_requests(batch_chunks)
        
        try:
            message_batch = client.beta.messages.batches.create(requests=batch_requests)
            batches.append(BatchStatus(
                batch_id=message_batch.id,
                chunks=batch_chunks,
                input_file=input_file  # Add input_file parameter
            ))
            msg = f"Submitted batch {message_batch.id} with {len(batch_chunks)} chunks for {input_file}"
            print(msg)
            notifier.send_notification("Batch Creation", msg, priority=5)
        except Exception as e:
            print(f"Error submitting batch for {input_file}: {str(e)}")
            raise
            
    return batches

    
def resume_from_batch(batch_id: str, output_path: str, client: anthropic.Anthropic, notifier: GotifyNotifier):
    """Resume processing from an existing batch ID."""
    try:
        # First verify the batch exists and get its status
        batch_status = client.beta.messages.batches.retrieve(batch_id)
        
        if batch_status.processing_status != "ended":
            raise ValueError(f"Batch {batch_id} is not completed (status: {batch_status.processing_status})")
        
        print(f"Found completed batch {batch_id}")
        notifier.send_notification(
            "Batch Resume Started",
            f"Starting to process results from batch {batch_id}",
            priority=5
        )
        
        # Initialize results dictionary
        results = {}
        error_count = 0
        
        # Process all results
        try:
            for result in client.beta.messages.batches.results(batch_id):
                chunk_id = result.custom_id
                if result.result.type == "succeeded":
                    message_content = result.result.message.content[0].text
                    results[chunk_id] = message_content
                else:
                    error_count += 1
                    error_type = result.result.type
                    error_details = getattr(result.result.error, 'message', 'No details available')
                    error_msg = (
                        f"Error in chunk {chunk_id}:\n"
                        f"Error Type: {error_type}\n"
                        f"Details: {error_details}"
                    )
                    print(error_msg)
                    notifier.send_notification(
                        f"Result Processing Error - Chunk {chunk_id}",
                        error_msg,
                        priority=7
                    )
        except Exception as e:
            error_msg = (
                f"Error processing batch results:\n"
                f"Error Type: {type(e).__name__}\n"
                f"Details: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            notifier.send_notification("Results Processing Error", error_msg, priority=9)
            raise
        
        if not results:
            raise ValueError("No successful results found in batch")
        
        # Sort and combine results
        sorted_results = sorted(results.items(), key=lambda x: int(x[0].split("_")[1]))
        processed_content = "\n".join(content for _, content in sorted_results)
        
        # Save processed content
        output_filepath = output_path.replace('.md', f'_from_batch_{batch_id}.md')
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        completion_msg = (
            f"Successfully processed {len(results)} chunks from batch {batch_id}\n"
            f"Encountered {error_count} errors\n"
            f"Output saved to: {output_filepath}"
        )
        
        notifier.send_notification(
            "Batch Resume Complete",
            completion_msg,
            priority=7
        )
        
        print(completion_msg)
        return output_filepath
        
    except Exception as e:
        error_msg = (
            f"Failed to resume from batch {batch_id}:\n"
            f"Error Type: {type(e).__name__}\n"
            f"Details: {str(e)}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        print(error_msg)
        notifier.send_notification("Batch Resume Error", error_msg, priority=10)
        raise
    
def process_files_in_parallel(input_path: str, client: anthropic.Anthropic, notifier: GotifyNotifier):
    """Process multiple markdown files in parallel using batch API"""
    files = get_text_files(input_path)
    if not files:
        raise ValueError(f"No text files found in {input_path}")
    
    processing_status = ProcessingStatus()
    last_notification_time = time.time()
    
    # Start processing each file
    for filepath in files:
        try:
            print(f"Starting processing of {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = chunk_markdown(content)
            if not chunks:
                raise ValueError("No chunks were created from the input file")
            
            # Pass notifier to submit_batches
            batches = submit_batches(chunks, client, filepath, notifier)
            processing_status.active_batches[filepath] = batches
            
            print(f"Submitted {len(batches)} batches for {filepath}")
        except Exception as e:
            error_msg = f"Failed to process {filepath}: {str(e)}"
            processing_status.failed_files[filepath] = error_msg
            notifier.send_notification(
                "File Processing Error",
                error_msg,
                priority=8
            )
    
    # Monitor all batches
    start_time = time.time()
    max_runtime = 24 * 60 * 60  # 24 hours in seconds
    
    while processing_status.active_batches:
        if time.time() - start_time > max_runtime:
            raise TimeoutError("Processing exceeded maximum runtime of 24 hours")
        
        # Send hourly status update
        current_time = time.time()
        if current_time - last_notification_time >= 3600:
            summary = get_batch_status_summary(processing_status, client)
            status_message = format_status_message(summary)
            notifier.send_notification(
                "Hourly Status Update",
                status_message,
                priority=5
            )
            last_notification_time = current_time
        
        # Check each file's batches
        for filepath in list(processing_status.active_batches.keys()):
            try:
                file_completed = True
                all_results = []
                
                for batch in processing_status.active_batches[filepath]:
                    if batch.status != "ended":
                        status = client.beta.messages.batches.retrieve(batch.batch_id)
                        if status.processing_status == "ended":
                            # Process batch results
                            try:
                                for result in client.beta.messages.batches.results(batch.batch_id):
                                    chunk_id = result.custom_id
                                    if result.result.type == "succeeded":
                                        message_content = result.result.message.content[0].text
                                        batch.results[chunk_id] = message_content
                                    else:
                                        error_msg = f"Error in chunk {chunk_id} of {filepath}: {result.result.type}"
                                        print(error_msg)
                                batch.status = "ended"
                            except Exception as e:
                                print(f"Error processing batch {batch.batch_id} results: {str(e)}")
                                file_completed = False
                        else:
                            file_completed = False
                
                if file_completed:
                    # Combine all batch results for this file
                    all_results = []
                    for batch in processing_status.active_batches[filepath]:
                        for chunk_id, content in batch.results.items():
                            all_results.append({"chunk_id": chunk_id, "content": content})
                    
                    # Sort and save results
                    all_results.sort(key=lambda x: int(x["chunk_id"].split("_")[1]))
                    processed_content = "\n".join(r["content"] for r in all_results)
                    
                    output_filepath = str(Path(filepath).with_suffix('')) + '_processed.md'
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write(processed_content)
                    
                    processing_status.completed_files.add(filepath)
                    del processing_status.active_batches[filepath]
                    
                    notifier.send_notification(
                        "File Processing Complete",
                        f"Successfully processed {filepath}",
                        priority=7
                    )
            except Exception as e:
                error_msg = f"Error processing {filepath}: {str(e)}"
                processing_status.failed_files[filepath] = error_msg
                del processing_status.active_batches[filepath]
                notifier.send_notification(
                    "File Processing Error",
                    error_msg,
                    priority=8
                )
        
        time.sleep(60)
    
    # Send final summary
    final_summary = {
        "total_files": len(files),
        "completed_files": len(processing_status.completed_files),
        "failed_files": len(processing_status.failed_files),
        "failures": processing_status.failed_files
    }
    
    notifier.send_notification(
        "Processing Complete",
        f"Processed {final_summary['completed_files']}/{final_summary['total_files']} files successfully",
        priority=7
    )
    
    return final_summary

def main():
    parser = argparse.ArgumentParser(description='Process markdown and text files using Anthropic API with notifications')
    parser.add_argument('--input', type=str, required=True, help='Input file or directory path')
    parser.add_argument('--dry-run', action='store_true', help='Test connections without processing')
    parser.add_argument('--gotify-url', type=str, help='Gotify server URL (overrides .env)')
    parser.add_argument('--gotify-token', type=str, help='Gotify notification token (overrides .env)')
    parser.add_argument('--api-key', type=str, help='Anthropic API key (overrides .env)')
    parser.add_argument('--resume-batch', type=str, help='Resume processing from an existing batch ID')
    args = parser.parse_args()

    # Load environment variables at the start
    load_dotenv(override=True)

    client = None
    notifier = None
    
    try:
        # Validate input file exists
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        # Initialize notifier first to ensure notifications work
        notifier = GotifyNotifier(
            base_url=args.gotify_url,  # Will use .env value if None
            token=args.gotify_token    # Will use .env value if None
        )

        # Test notification immediately
        if not notifier.send_notification(
            "Script Started",
            f"Starting processing of files in {args.input}",
            priority=5
        ):
            raise ValueError("Failed to send initial notification - check Gotify settings")
            
        # Setup API key
        api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("No API key provided. Set ANTHROPIC_API_KEY in .env file or pass via --api-key")
        
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)

        if args.dry_run:
            print("Running in dry-run mode...")
            if test_connections(client, notifier):
                print("All connections successful!")
                notifier.send_notification(
                    "Connection Test Successful",
                    "All required connections are working properly.",
                    priority=5
                )
            else:
                error_msg = "Connection tests failed! Check individual connection error messages above."
                print(error_msg)
                notifier.send_notification(
                    "Connection Test Failed",
                    error_msg,
                    priority=9
                )
            return

        if args.resume_batch:
            output_file = resume_from_batch(args.resume_batch, args.input, client, notifier)
            print(f"Resume processing complete. Output saved to: {output_file}")
        else:
            summary = process_files_in_parallel(args.input, client, notifier)
            print(f"Processing complete. Successfully processed {summary['completed_files']}/{summary['total_files']} files")
            if summary['failed_files']:
                print("\nFailed files:")
                for filepath, error in summary['failures'].items():
                    print(f"- {filepath}: {error}")
                    
            # Send final notification with summary
            final_msg = (
                f"Processing complete for {args.input}\n"
                f"Successfully processed: {summary['completed_files']}/{summary['total_files']} files\n"
                f"Failed files: {summary['failed_files']}\n"  # failed_files is already a count in the summary dict
            )
            if summary['failures']:  # Use 'failures' key for the dict of failed files
                final_msg += "\nFailed files:\n" + "\n".join(
                    f"- {filepath}: {error}" 
                    for filepath, error in summary['failures'].items()
                )
            notifier.send_notification(
                "Processing Complete",
                final_msg,
                priority=7
            )
            
    except Exception as e:
        error_msg = (
            f"Fatal error during script execution:\n"
            f"Error Type: {type(e).__name__}\n"
            f"Details: {str(e)}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        print(error_msg)
        
        # Try to send notification even if notifier isn't fully initialized
        if notifier:
            try:
                notifier.send_notification("Fatal Error", error_msg, priority=10)
            except Exception as notify_error:
                print(f"Additionally failed to send error notification: {str(notify_error)}")
        
        sys.exit(1)  # Exit with error code
    finally:
        # Send final notification
        if notifier:
            try:
                notifier.send_notification(
                    "Script Completed",
                    "Processing has finished (success or failure).",
                    priority=5
                )
            except:
                pass
            
        # Cleanup if needed
        if client:
            pass  # Any cleanup needed for the client

if __name__ == "__main__":
    main()