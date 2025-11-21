"""
Quick Setup Test Script
=======================
Run this to verify your environment is properly configured for the interview.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor} (Need 3.8+)")
        return False


def check_imports():
    """Check if all required packages are installed"""
    print("\nChecking package installations...")
    
    packages = {
        "langchain": "langchain",
        "langchain_openai": "langchain-openai",
        "langchain_qdrant": "langchain-qdrant",
        "openai": "openai",
        "pydantic": "pydantic",
        "qdrant_client": "qdrant-client",
        "tiktoken": "tiktoken",
    }
    
    all_good = True
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Run: pip install {package}")
            all_good = False
    
    return all_good


def check_api_key():
    """Check if OpenAI API key is set"""
    print("\nChecking API key configuration...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("sk-"):
        print(f"  ✅ OPENAI_API_KEY is set")
        return True
    else:
        print(f"  ❌ OPENAI_API_KEY not set or invalid")
        print(f"     Set with: export OPENAI_API_KEY='your-key-here'")
        return False


def test_openai_connection():
    """Test actual connection to OpenAI"""
    print("\nTesting OpenAI API connection...")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'API working!'"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"  ✅ API Connection successful")
        print(f"     Response: {result}")
        return True
        
    except Exception as e:
        print(f"  ❌ API Connection failed: {e}")
        return False


def test_langchain():
    """Test LangChain basic functionality"""
    print("\nTesting LangChain setup...")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_template("Say '{word}'")
        chain = prompt | llm
        
        result = chain.invoke({"word": "LangChain works!"})
        print(f"  ✅ LangChain working")
        print(f"     Response: {result.content}")
        return True
        
    except Exception as e:
        print(f"  ❌ LangChain test failed: {e}")
        return False


def test_embeddings():
    """Test embeddings and Qdrant"""
    print("\nTesting embeddings and Qdrant...")
    
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        
        embeddings = OpenAIEmbeddings()
        texts = ["Test document 1", "Test document 2"]
        vectorstore = QdrantVectorStore.from_texts(
            texts,
            embeddings,
            location=":memory:",
            collection_name="test"
        )
        
        results = vectorstore.similarity_search("Test", k=1)
        print(f"  ✅ Embeddings and Qdrant working")
        print(f"     Retrieved: {results[0].page_content}")
        return True
        
    except Exception as e:
        print(f"  ❌ Embeddings test failed: {e}")
        return False


def test_pydantic_parser():
    """Test Pydantic output parser"""
    print("\nTesting Pydantic structured outputs...")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field
        
        class TestOutput(BaseModel):
            message: str = Field(description="A test message")
            number: int = Field(description="A test number")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(TestOutput)
        
        prompt = ChatPromptTemplate.from_template(
            "Return message='Hello' and number=42"
        )
        
        chain = prompt | structured_llm
        result = chain.invoke({})
        
        print(f"  ✅ Structured output working")
        print(f"     Parsed: message='{result.message}', number={result.number}")
        return True
        
    except Exception as e:
        print(f"  ❌ Structured output test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("iBusiness Funding Interview - Setup Verification")
    print("="*70)
    
    tests = [
        ("Python Version", check_python_version),
        ("Package Installations", check_imports),
        ("API Key", check_api_key),
        ("OpenAI Connection", test_openai_connection),
        ("LangChain", test_langchain),
        ("Embeddings & Qdrant", test_embeddings),
        ("Structured Output", test_pydantic_parser),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} - {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All systems ready for interview! Good luck!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nQuick fixes:")
        print("  1. Install packages: pip install -r requirements.txt")
        print("  2. Set API key: export OPENAI_API_KEY='your-key'")
        print("  3. Verify Python 3.8+: python --version")
        return 1


if __name__ == "__main__":
    exit(main())

