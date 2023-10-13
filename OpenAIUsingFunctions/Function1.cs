using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;

namespace OpenAIUsingFunctions
{
    public static class Functions1
    {
        [FunctionName("Summariser")]
        public static async Task<IActionResult> Summariser(
            [HttpTrigger(AuthorizationLevel.Anonymous, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string instruction = req.Query["instruction"];
            string searchTerm = req.Query["searchTerm"];
            string ask = req.Query["ask"];
            string tone = req.Query["tone"];
            string topN = req.Query["topn"];
            string maxOutputTokens = req.Query["maxOutputTokens"];
            string temperature = req.Query["temperature"];
            //string reserveTokens = req.Query["reserveTokens"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);

            instruction = instruction?? data?.instruction;  //You are an AI assistant that helps people find information. Only answer questions using the sources below and if you cannot find the answer than say 'I don't know'.
            searchTerm = searchTerm ?? data?.searchTerm;
            ask = ask ?? data?.ask;
            tone = tone ?? data?.tone;
            topN = topN ?? data?.topn;
            maxOutputTokens = maxOutputTokens ?? data?.maxOutputTokens;
            temperature = temperature ?? data?.temperature;
           // reserveTokens = reserveTokens ?? data?.reserveTokens;

            int intTopNResults = 0;
            float floatTemperature = 0;
            int intMaxOutputTokens = 0;
            
            int.TryParse(topN, out intTopNResults);
            int.TryParse(maxOutputTokens, out intMaxOutputTokens);
            float.TryParse(temperature, out floatTemperature);
            
            intTopNResults = intTopNResults == 0 ? 1 : intTopNResults;
            intMaxOutputTokens = intMaxOutputTokens == 0 ? 300 : intMaxOutputTokens;
            

            //Cognitive search settings
            CognitiveSearchSettings cogSettings = new CognitiveSearchSettings();
            cogSettings.ServiceEndPoint = "<cogendpoint>";
            cogSettings.SearchQueryApiKey = "cogsearchkey>";
            cogSettings.SearchIndexName = "<cogsearchindex>";
            cogSettings.TopNResults = intTopNResults;

            //OpenAI settings
            OpenAISettings openAISettings = new OpenAISettings();
            openAISettings.ServiceEndpoint = "<openAIendpoint>";
            openAISettings.ApiKey = "<openaikey>";

            //openAISettings.DeploymentId = "gpt35turbo";
            //openAISettings.GPTModel = GPTModel.GPT35Turbo;
            openAISettings.DeploymentId = "gpt35turbo16k"; //works with only chat completions
            openAISettings.GPTModel = GPTModel.GPT35Turbo16k;

            openAISettings.Tone = tone;
            openAISettings.Temperature = floatTemperature;
            openAISettings.MaxOutputTokens = intMaxOutputTokens;
            
            openAISettings.Instruction = instruction;

            OpenAIImplementation openAI = new OpenAIImplementation(cogSettings, openAISettings, ask);

            var result = await openAI.ProcessRequest(searchTerm);

            return new OkObjectResult(result);
        }
    }
}
