update the docs with the recent work and ALL the changes in git. Ensure documentation complies with the documentation standards and expectations outlined in the claude.md file. Then commit.




- we recently made the ws_paper_tester/ws_tester/indicators/ section of ws_paper_tester/ I want develop and add a algorithm to track the token pairs(XRP/USDT, BTC/USDT, XRP/BTC) and overall market direction. Like a bear, bull, or flat/sideways detector that the strategies can use in their evals. Maybe strategies can have different parameters based on the algorithms output to maximize it effectiveness. deep research this idea, ultrathink a plan, and make documentation of your findings and recommendations. be sure to consider what data is available to us via the full kraken api. We can add other free services. Put your documentation in ws_paper_tester/docs/development/plans/b-b-detector/
- what historic data is available via the kraken api? should we make a database of historic data for backtesting? we have a local psql or we can make a docker psql we can use
- do we have any algos the use margin/leverage?


we just added a huge feature ws_paper_tester/docs/development/plans/b-b-detector/ and integrated it into our strategies. We need to do a deep code and logic review of the additions, changes, and integrations. ultrathink