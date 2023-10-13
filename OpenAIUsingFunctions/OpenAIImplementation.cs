using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Azure;
using Azure.Search.Documents.Models;
using Azure.Search.Documents;
using Newtonsoft.Json;
using System.Security.Principal;
using static System.Net.Mime.MediaTypeNames;
using System.Text.RegularExpressions;
using Azure.AI.OpenAI;
using Microsoft.ML.Tokenizers;
using SharpToken;
using System.Text.Json.Serialization;

//https://github.com/Azure-Samples/openai-dotnet-samples
//https://learn.microsoft.com/en-us/dotnet/api/azure.ai.openai?view=azure-dotnet-preview

namespace OpenAIUsingFunctions
{
    public class OpenAIImplementation
    {
        private CognitiveSearchSettings cogSettings;
        private OpenAISettings openAISettings;
        private GptEncoding gptEncoding;
        private int remainingTokens = 0;
        public int maxOpenAITokens = 0;
        private StringBuilder openAIPrompt;
        private GPTModel GPTModel;

        public string Ask { get; private set; } //question from user

        public OpenAIImplementation(CognitiveSearchSettings cogSettings, OpenAISettings openAISettings, string ask)
        {
            this.gptEncoding = GptEncoding.GetEncodingForModel("gpt-4");
            this.cogSettings = cogSettings;
            this.openAISettings = openAISettings;
            this.openAIPrompt = new StringBuilder();

            this.Ask = ask;


            switch (this.openAISettings.GPTModel)
            {
                case GPTModel.GPT35Turbo:
                    this.maxOpenAITokens = 4000;
                    break;

                case GPTModel.GPT35Turbo16k:
                    this.maxOpenAITokens = 16000;
                    break;

                case GPTModel.GPT4:
                    this.maxOpenAITokens = 8000;
                    break;

                case GPTModel.GPT432K:
                    this.maxOpenAITokens = 32000;
                    break;

                default:
                    this.maxOpenAITokens = 4000;
                    break;
            }

            //calculate ask tokens
            var askTokens = this.gptEncoding.Encode(this.Ask).Count;

            this.remainingTokens = this.maxOpenAITokens - askTokens - this.openAISettings.MaxOutputTokens;
        }

        public async Task<OpenAIResult> ProcessRequest(string searchTerm)
        {
            //Perform cognitive search
            var docsamples = PerformCogSearch(searchTerm);

            ////calculate ask tokens
            //var askTokens = this.gptEncoding.Encode(ask).Count;

            ////adjust tokens
            //this.remainingTokens = this.remainingTokens - askTokens;

            //Call Open AI
            OpenAIResult result;
            GenerateOpenAIPrompt(docsamples); //generate open instruction and prompt

            //result = CallOpenAICompletions(); //completions

            result = await CallOpenAIChatCompletions(); //chat completions

            return result;
        }

        private async Task<OpenAIResult> CallOpenAIChatCompletions()
        {
            OpenAIClient openAIClient = new OpenAIClient(new Uri(this.openAISettings.ServiceEndpoint), new Azure.AzureKeyCredential(this.openAISettings.ApiKey));
            OpenAIResult result = new OpenAIResult();

            ChatCompletionsOptions chatCompletionsOptions = new ChatCompletionsOptions
            {
                Messages =
                {
                    new ChatMessage(ChatRole.System, this.openAIPrompt.ToString()),
                    new ChatMessage(ChatRole.User, this.Ask)
                },
                Temperature = openAISettings.Temperature,
                MaxTokens = this.openAISettings.MaxOutputTokens,
                NucleusSamplingFactor = (float)0,
                FrequencyPenalty = (float)0,
                PresencePenalty = (float)0,
            };

            Response<ChatCompletions> response = await openAIClient.GetChatCompletionsAsync(this.openAISettings.DeploymentId, chatCompletionsOptions);

            result.AOAIPrompt = this.openAIPrompt.ToString();

            if (response.Value.Choices != null && response.Value.Choices.Count > 0)
            {
                result.AOAIResponse = response.Value.Choices[0].Message.Content.Trim();
            }

            return result;
        }

        private async Task<OpenAIResult> CallOpenAICompletions()
        {
            OpenAIClient openAIClient = new OpenAIClient(new Uri(this.openAISettings.ServiceEndpoint), new Azure.AzureKeyCredential(this.openAISettings.ApiKey));
            OpenAIResult result = new OpenAIResult();

            CompletionsOptions completionOptions = new CompletionsOptions
            {
                MaxTokens = this.openAISettings.MaxOutputTokens,
                Temperature = openAISettings.Temperature,
                NucleusSamplingFactor = (float)0,
                FrequencyPenalty = (float)0,
                PresencePenalty = (float)0,
                Prompts = { this.openAIPrompt.ToString() }

            };


            Response<Completions> completionResponse = await openAIClient.GetCompletionsAsync(this.openAISettings.DeploymentId, completionOptions);

            result.AOAIPrompt = this.openAIPrompt.ToString();

            if (completionResponse.Value.Choices != null && completionResponse.Value.Choices.Count > 0)
            {
                result.AOAIResponse = completionResponse.Value.Choices[0].Text.Trim();
            }

            return result;
        }

        private StringBuilder GenerateOpenAIPrompt(DocSample[] docSamples)
        {
            //instruction
            //this.openAIPrompt.AppendLine("You are an AI assistant that helps people find information. Only answer questions using the sources below and if you cannot find the answer than say 'I don't know'.");
            this.openAIPrompt.AppendLine(this.openAISettings.Instruction);
            this.openAIPrompt.AppendLine();

            //if (!string.IsNullOrWhiteSpace(ask))
            //{
            //    this.openAIPrompt.Append(ask);
            //}

            if (!string.IsNullOrEmpty(this.openAISettings.Tone))
            {
                //this.openAIPrompt.AppendLine();
                openAIPrompt.Append("\nPlease answer the question in the tone of voice of" + this.openAISettings.Tone);

            }

            DocSample doc;

            for (int i = 0; i < docSamples.Length; i++)
            {
                doc = docSamples[i];

                TruncateTextForOpenAI(doc.content, i + 1);
            }

            return this.openAIPrompt;
        }

        public void TruncateTextForOpenAI(string text, int sourceIndex)
        {
            int tokenCount = -1;

            text = RemoveEmptyLines(text); //remove empty lines

            //text = $"Source {sourceIndex}: {text}\r\n"; //append source index. \n for breaking sources

            var splitText = SplitText(text); //chunnk text

            splitText.Insert(0, $"\r\n\r\nSource {sourceIndex}:\r\n");

            //splitText[0] = "\r\n\r\n" + splitText[0];


            foreach (var str in splitText)
            {
                tokenCount = this.gptEncoding.Encode(this.openAIPrompt.ToString()).Count;

                if (tokenCount < this.remainingTokens)
                {
                    tokenCount = this.gptEncoding.Encode(this.openAIPrompt.ToString() + str).Count;

                    if (tokenCount < this.remainingTokens)
                    {
                        this.openAIPrompt.Append(str);
                        //this.remainingTokens = this.openAISettings.MaxTokens - tokenCount;
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }
            }

            // return finalText.ToString();
        }

        private string RemoveEmptyLines(string str)
        {
            return Regex.Replace(str, @"^\s*$\n|\r", string.Empty, RegexOptions.Multiline).TrimEnd();
        }

        private List<string> SplitText(string text)
        {
            int maxChunkSize = 1000;
            List<string> chunks = new List<string>();
            string currentChunk = "";

            foreach (string sentence in text.Split('.'))
            {
                if (currentChunk.Length + sentence.Length < maxChunkSize)
                {
                    currentChunk += sentence + ".";
                }
                else
                {
                    chunks.Add(currentChunk.Trim());
                    currentChunk = sentence + ".";
                }
            }

            if (!string.IsNullOrEmpty(currentChunk))
            {
                chunks.Add(currentChunk.Trim());
            }

            return chunks;
        }

        private DocSample[] PerformCogSearch(string searchTerm)
        {
            SearchClient searchClient = new SearchClient(new Uri(this.cogSettings.ServiceEndPoint),
                this.cogSettings.SearchIndexName, new AzureKeyCredential(this.cogSettings.SearchQueryApiKey));


            SearchOptions options;
            SearchResults<DocSample> results;
            //SearchResults<SearchDocument> results;


            options = new SearchOptions();
            options.Select.Add("content");
            options.SearchFields.Add("content");
            options.QueryType = SearchQueryType.Simple;
            options.Size = this.cogSettings.TopNResults;

            //results = searchClient.Search<SearchDocument>(searchTerm, options);
            results = searchClient.Search<DocSample>(searchTerm, options);

            return results.GetResults().Select(a => a.Document).ToArray();
        }
    }

    public class CognitiveSearchSettings
    {
        public string ServiceEndPoint { get; set; }
        public string SearchQueryApiKey { get; set; }
        public string SearchIndexName { get; set; }
        public int TopNResults { get; set; }
    }

    public class OpenAISettings
    {
        public string Instruction { get; set; }
        public string Tone { get; set; }

        public string DeploymentId { get; set; }

        public string ApiKey { get; set; }

        public string ServiceEndpoint { get; set; }

        public int MaxOutputTokens { get; set; }
        public float Temperature { get; set; }

        //public int ReserveTokens { get; set; }

        public GPTModel GPTModel { get; set; }
    }

    public class DocSample
    {
        public string content { get; set; }

        [System.Text.Json.Serialization.JsonPropertyName("@search.score")]
        public double score { get; set; }
    }

    public class OpenAIResult
    {
        [Newtonsoft.Json.JsonProperty(Order = 2)]
        public string AOAIPrompt { get; set; }

        [Newtonsoft.Json.JsonProperty(Order = 1)]
        public string AOAIResponse { get; set; }
    }

    public enum GPTModel
    {
        GPT35Turbo = 1,
        GPT35Turbo16k = 2,
        GPT4 = 3,
        GPT432K = 4
    }

}
