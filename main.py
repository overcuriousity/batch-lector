import anthropic
import sys
import time
import os
import requests
import argparse
import traceback
from typing import List, Dict, Set
from dataclasses import dataclass
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request


@dataclass
class BatchStatus:
    batch_id: str
    chunks: List[Dict[str, str]]
    status: str = "in_progress"
    results: Dict[str, str] = None

    def __init__(self, batch_id: str, chunks: List[Dict[str, str]], status: str = "in_progress"):
        self.batch_id = batch_id
        self.chunks = chunks
        self.status = status
        self.results = {} if self.results is None else self.results
    

class GotifyNotifier:
    def __init__(self, base_url: str = "https://push.example.de", token: str = "AQk4vAALOrMNRJi"):
        self.base_url = base_url.rstrip('/')
        self.token = token
        
    def send_notification(self, title: str, message: str, priority: int = 5):
        """Send notification via Gotify"""
        url = f"{self.base_url}/message"
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
    system_prompt = """You are an expert at fixing markdown formatting while preserving the original content. 
    Fix only structural markdown issues, hyphenation errors, and formatting. DO NOT change or translate the German text content.
    Fix all original line breaks and paragraph structures. Return only the fixed markdown without explanations or annotations. Remove any Footnotes which might interfere with the text. Don´t change any image references."""
    
    return [
        Request(
            custom_id=f"chunk_{i}_{chunk['start_line']}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=4096,
                system=system_prompt,  # System prompt as top-level parameter
                messages=[
                    {"role": "user", "content": chunk['content']}
                ]
            )
        )
        for i, chunk in enumerate(chunks)
    ]

def submit_batches(chunks: List[Dict[str, str]], client: anthropic.Anthropic, max_batch_size: int = 10000) -> List[BatchStatus]:
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
                chunks=batch_chunks
            ))
            print(f"Submitted batch {message_batch.id} with {len(batch_chunks)} chunks")
        except Exception as e:
            print(f"Error submitting batch: {str(e)}")
            raise
            
    return batches

def get_batch_status_summary(batches: List[BatchStatus], client: anthropic.Anthropic) -> Dict:
    """Get a summary of current batch processing status."""
    status_counts = {
        "total": len(batches),
        "completed": 0,
        "in_progress": 0,
        "error": 0
    }
    
    request_counts = {
        "total": 0,
        "processing": 0,
        "succeeded": 0,
        "errored": 0,
        "canceled": 0,
        "expired": 0
    }
    
    for batch in batches:
        if batch.status == "ended":
            status_counts["completed"] += 1
        else:
            try:
                status = client.beta.messages.batches.retrieve(batch.batch_id)
                if status.processing_status == "ended":
                    status_counts["completed"] += 1
                else:
                    status_counts["in_progress"] += 1
                
                # Update request counts WITHOUT double counting total
                for count_type, count in status.request_counts.items():
                    request_counts[count_type] += count
                request_counts["total"] = sum(request_counts[k] for k in ["processing", "succeeded", "errored", "canceled", "expired"])
            except Exception:
                status_counts["error"] += 1
    
    return {
        "batch_status": status_counts,
        "request_counts": request_counts
    }

def check_batch_statuses(batches: List[BatchStatus], client: anthropic.Anthropic, notifier: GotifyNotifier, 
                        last_notification_time: float) -> tuple[Set[str], float]:
    """Check status of all batches and collect results from completed ones."""
    completed_batch_ids = set()
    current_time = time.time()
    
    # Send hourly status update
    if current_time - last_notification_time >= 3600:
        status_summary = get_batch_status_summary(batches, client)
        batch_stats = status_summary["batch_status"]
        req_stats = status_summary["request_counts"]
        
        status_message = (
            f"Batch Processing Status:\n"
            f"Batches: {batch_stats['completed']}/{batch_stats['total']} completed\n"
            f"Requests: {req_stats['succeeded']}/{req_stats['total']} succeeded\n"
            f"In Progress: {req_stats['processing']} requests\n"
            f"Errors: {req_stats['errored']} requests\n"
            f"Canceled/Expired: {req_stats['canceled'] + req_stats['expired']} requests"
        )
        
        notifier.send_notification(
            "Hourly Status Update",
            status_message,
            priority=5
        )
        last_notification_time = current_time
    
    # Check individual batch status
    for batch in batches:
        if batch.status != "ended":
            try:
                status = client.beta.messages.batches.retrieve(batch.batch_id)
                if status.processing_status == "ended":
                    error_count = 0
                    # Collect results
                    try:
                        for result in client.beta.messages.batches.results(batch.batch_id):
                            # Access result attributes using dot notation instead of dictionary access
                            chunk_id = result.custom_id
                            if result.result.type == "succeeded":
                                # Access the content correctly based on the API response structure
                                message_content = result.result.message.content[0].text
                                batch.results[chunk_id] = message_content
                            else:
                                error_count += 1
                                error_type = result.result.type
                                error_details = getattr(result.result.error, 'message', 'No details available')
                                error_msg = (
                                    f"Error processing chunk {chunk_id} in batch {batch.batch_id}\n"
                                    f"Error Type: {error_type}\n"
                                    f"Details: {error_details}\n"
                                    f"Start Line: {chunk_id.split('_')[-1]}"
                                )
                                print(error_msg)
                                notifier.send_notification(
                                    f"Processing Error - Chunk {chunk_id}",
                                    error_msg,
                                    priority=8
                                )
                    except Exception as e:
                        error_msg = (
                            f"Error processing results for batch {batch.batch_id}:\n"
                            f"Error Type: {type(e).__name__}\n"
                            f"Details: {str(e)}"
                        )
                        print(error_msg)
                        notifier.send_notification(
                            "Results Processing Error",
                            error_msg,
                            priority=9
                        )
                        continue
                    
                    # Send batch completion notification with error summary
                    if error_count > 0:
                        notifier.send_notification(
                            f"Batch {batch.batch_id} Completed with Errors",
                            f"Batch completed with {error_count} failed chunks out of {len(batch.chunks)}",
                            priority=7
                        )
                    
                    batch.status = "ended"
                    completed_batch_ids.add(batch.batch_id)
                    print(f"Batch {batch.batch_id} completed")
            except Exception as e:
                error_msg = (
                    f"Error checking batch {batch.batch_id}:\n"
                    f"Error Type: {type(e).__name__}\n"
                    f"Details: {str(e)}"
                )
                print(error_msg)
                notifier.send_notification(
                    "Batch Status Check Error",
                    error_msg,
                    priority=9
                )
                
    return completed_batch_ids, last_notification_time

def process_markdown_in_parallel(filepath: str, client: anthropic.Anthropic, notifier: GotifyNotifier):
    """Process markdown file by submitting all chunks in parallel batches."""
    output_filepath = None
    try:
        # Read markdown file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        error_msg = (
            f"Failed to read input file: {filepath}\n"
            f"Error Type: {type(e).__name__}\n"
            f"Details: {str(e)}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        notifier.send_notification("File Read Error", error_msg, priority=10)
        raise

    try:
        # Split content into chunks
        chunks = chunk_markdown(content)
        if not chunks:
            raise ValueError("No chunks were created from the input file")
            
        total_chunks = len(chunks)
        print(f"Split content into {total_chunks} chunks")
        notifier.send_notification(
            "Processing Started",
            f"Started processing markdown file with {total_chunks} chunks",
            priority=7
        )
        
        # Submit all chunks in batches
        batches = submit_batches(chunks, client)
        if not batches:
            raise ValueError("No batches were created from the chunks")
            
        total_batches = len(batches)
        print(f"Submitted {total_batches} batches")
    
        # Initialize status tracking
        completed_batches = set()
        last_notification_time = time.time()
        
        # Send initial status
        status_summary = get_batch_status_summary(batches, client)
        initial_status = (
            f"Initial Processing Status:\n"
            f"Total Batches: {status_summary['batch_status']['total']}\n"
            f"Total Requests: {status_summary['request_counts']['total']}\n"
            f"Processing Started: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        notifier.send_notification("Processing Started", initial_status, priority=7)
        
        # Monitor batch completion with timeout
        start_time = time.time()
        max_runtime = 24 * 60 * 60  # 24 hours in seconds
        
        while len(completed_batches) < total_batches:
            if time.time() - start_time > max_runtime:
                raise TimeoutError("Processing exceeded maximum runtime of 24 hours")
                
            newly_completed, last_notification_time = check_batch_statuses(
                batches, client, notifier, last_notification_time
            )
            completed_batches.update(newly_completed)
            
            if len(completed_batches) < total_batches:
                print(f"Completed {len(completed_batches)}/{total_batches} batches...")
                time.sleep(60)
        
        # Combine results from all batches
        all_results = []
        for batch in batches:
            if not batch.results:
                raise ValueError(f"Batch {batch.batch_id} has no results")
            for chunk_id, content in batch.results.items():
                all_results.append({"chunk_id": chunk_id, "content": content})
        
        if not all_results:
            raise ValueError("No results were collected from completed batches")
        
        # Sort and combine results
        all_results.sort(key=lambda x: int(x["chunk_id"].split("_")[1]))
        processed_content = "\n".join(r["content"] for r in all_results)
        
        # Save processed content
        output_filepath = filepath.replace('.md', '_processed.md')
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        notifier.send_notification(
            "Processing Complete",
            f"Successfully processed {len(all_results)} chunks from {total_batches} batches. Output saved to: {output_filepath}",
            priority=7
        )
        
        return output_filepath

    except Exception as e:
        error_msg = (
            f"Failed during batch processing\n"
            f"Error Type: {type(e).__name__}\n"
            f"Details: {str(e)}\n"
            f"Traceback: {traceback.format_exc()}\n"
            f"Stage: {'chunk creation' if 'batches' not in locals() else 'batch submission'}"
        )
        notifier.send_notification("Processing Error", error_msg, priority=10)
        
        # Try to save partial results if available
        if 'all_results' in locals() and all_results:
            try:
                partial_output = filepath.replace('.md', '_partial_processed.md')
                with open(partial_output, 'w', encoding='utf-8') as f:
                    f.write("\n".join(r["content"] for r in all_results))
                notifier.send_notification(
                    "Partial Results Saved",
                    f"Saved partial results to: {partial_output}",
                    priority=7
                )
            except Exception as save_error:
                notifier.send_notification(
                    "Failed to Save Partial Results",
                    f"Error: {str(save_error)}",
                    priority=8
                )
        raise
    
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

def main():
    parser = argparse.ArgumentParser(description='Process markdown files using Anthropic API with notifications')
    parser.add_argument('--input', type=str, required=True, help='Input markdown file path')
    parser.add_argument('--dry-run', action='store_true', help='Test connections without processing')
    parser.add_argument('--gotify-url', type=str, default='https://push.example.de', help='Gotify server URL')
    parser.add_argument('--gotify-token', type=str, help='Gotify notification token')
    parser.add_argument('--api-key', type=str, help='Anthropic API key (alternatively use ANTHROPIC_API_KEY env variable)')
    parser.add_argument('--resume-batch', type=str, help='Resume processing from an existing batch ID')
    args = parser.parse_args()

    client = None
    notifier = None
    
    try:
        # Validate input file exists
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
            
        # Setup API key
        api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("No API key provided. Either set ANTHROPIC_API_KEY environment variable or use --api-key argument")
        
        # Initialize clients
        client = anthropic.Anthropic(api_key=api_key)
        notifier = GotifyNotifier(
            base_url=args.gotify_url,
            token=args.gotify_token or os.getenv('GOTIFY_TOKEN')
        )

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
        else:
            output_file = process_markdown_in_parallel(args.input, client, notifier)
            
        print(f"Processing complete. Output saved to: {output_file}")
        
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
        # Cleanup if needed
        if client:
            # Any cleanup needed for the client
            pass

if __name__ == "__main__":
    main()