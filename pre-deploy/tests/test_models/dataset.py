"""
Establishes a dataset fixture for testing the determinism and consistency
of the accuracy_with_reference metric, particularly in handling variations in
formatting, wording, and structure while maintaining key factual accuracy.
The dataset includes a range of cases such as exact matches, paraphrases,
omissions, contradictions, harmless elaborations, hallucinated additions,
numeric contradictions, vague but correct responses, multi-fact omissions,
and formatting differences.
"""

import time

from pre_deploy.input import EvalDataset, Conversation, Message, MessageContent


def _sampled_conversation(
    conversation_id: str,
    prompt: str,
    expected_output: str,
    actual_output: str,
    source_row: int,
    source_evaluate: int,
    source_rand_selection: int,
    source_thread_ts: str,
) -> Conversation:
    return Conversation(
        id=conversation_id,
        messages=[
            Message(
                sequence=0,
                role="user",
                contents=[MessageContent(type="text", content=prompt)],
            ),
            Message(
                sequence=1,
                role="assistant",
                contents=[MessageContent(type="text", content=actual_output)],
            ),
        ],
        metadata={
            "expected_output": expected_output,
            "source_row": source_row,
            "source_evaluate": source_evaluate,
            "source_rand_selection": source_rand_selection,
            "source_thread_ts": source_thread_ts,
        },
    )


dataset_fixture = EvalDataset(
    id=f"test_accuracy_dataset_{time.time()}",
    metadata={
        "dataset_version_id": "unit-test-edge-cases-v2",
        "purpose": "Probe accuracy_with_reference boundary behavior",
        "source_file": "sampled_qas.xlsx",
    },
    conversations=[
        _sampled_conversation(
            conversation_id="exact_match",
            prompt="Where can I find the project documentation?",
            expected_output="The project documentation is available at https://docs.example.com/projects/overview.",
            actual_output="The project documentation is available at https://docs.example.com/projects/overview.",
            source_row=2,
            source_evaluate=0,
            source_rand_selection=9,
            source_thread_ts="1724775891.77346",
        ),
        _sampled_conversation(
            conversation_id="semantic_equivalence",
            prompt="Where can I find the table to pull user data from the platform?",
            expected_output="The table to pull user data can be found in the Data Platform schema documentation at https://docs.example.com/schema. Specifically, the user_dim table is recommended for this purpose.",
            actual_output="Check the Data Platform schema documentation at https://docs.example.com/schema; the user_dim table is the recommended table for pulling user data from the platform.",
            source_row=47,
            source_evaluate=1,
            source_rand_selection=9,
            source_thread_ts="1600883354.003",
        ),
        _sampled_conversation(
            conversation_id="omission",
            prompt="Where can I find comparison reports for Q1 2021 vs Q1 2022?",
            expected_output="The reporting site only provides options for Q1 2022 vs Q1 2023. For a 5-year analysis, it is recommended to use the quarterly census data for trend reporting. The tracking reports are not based on quarterly snapshots but on point-in-time comparisons. Custom reports can be built using snapshots going back to 2005, but a specific point in time must be chosen for comparison. It is suggested to check the reports published by the analytics team.",
            actual_output="For a longer-term comparison, use the quarterly census data for trend reporting.",
            source_row=14,
            source_evaluate=2,
            source_rand_selection=5,
            source_thread_ts="1676322629.92399",
        ),
        _sampled_conversation(
            conversation_id="contradiction",
            prompt="Is there a report that will pull all premium users enrolled going back to Summer 2018 in a specific department?",
            expected_output="The enrollment reports only provide data for current terms. However, data going back to 2010 can be accessed in the archive system. A report that breaks down enrollment by department exists but only includes fall and spring terms and provides aggregate data, which does not meet the specific need for user lists. There is a possibility to modify an existing query to meet the requirements, but further discussions are needed.",
            actual_output="Yes, the reports provide summer user lists going back to Summer 2018 for specific departments.",
            source_row=24,
            source_evaluate=0,
            source_rand_selection=3,
            source_thread_ts="1594322835.1421",
        ),
        _sampled_conversation(
            conversation_id="harmless_elaboration_in_expected_output",
            prompt="How can I find a report to determine the number of users in a program and new signups for Q4 2024?",
            expected_output="To find the desired report, check the User lists on the Enrollment and Records page at https://reports.example.com/enrollment/pages/welcome. The report titled \"Users by Plan as of Today\" lists all active users in a program, including their signup type. You can also run this report by program or plan code. For new users, consider the report at https://reports.example.com/enrollment/new-users. Additionally, the User Enrollment by Department report at https://reports.example.com/enrollment/by-department allows for drilling down by department, providing detailed enrollment data.",
            actual_output="Start with the User lists page on Enrollment and Records, then use 'Users by Plan as of Today' to count users in a program and identify new signups. You can also drill into the User Enrollment by Department report for more detail.",
            source_row=3,
            source_evaluate=7,
            source_rand_selection=0,
            source_thread_ts="1724089308.33667",
        ),
        _sampled_conversation(
            conversation_id="hallucinated_addition_in_actual_output",
            prompt="Does anyone know of a report that shows older enrollment data for a specific program? I need data from Fall 2021.",
            expected_output="It is suggested to use the Enterprise group page, particularly the Master List for helpful information. For census enrollment data, refer to the following link: https://reports.example.com/institutional-analysis/enrollment",
            actual_output="It is suggested to use the Enterprise group page, particularly the Master List for helpful information. For census enrollment data, refer to the following link: https://reports.example.com/institutional-analysis/enrollment. This report also includes department-level historical trends back to 1998.",
            source_row=4,
            source_evaluate=8,
            source_rand_selection=0,
            source_thread_ts="1709830802.35589",
        ),
        _sampled_conversation(
            conversation_id="numeric_contradiction",
            prompt="Is there a report anywhere that shows users registered for services by department for spring 2023?",
            expected_output="Yes, there are reports available that can be run on demand by department, plan, region, and tier. For a list of users, visit the User Lists page at https://reports.example.com/enrollment/pages/users. For counts by plan, refer to the Enrollment Tracking page at https://reports.example.com/enrollment/pages/tracking. Note that Enrollment Tracking only counts users registered in standard classes, excluding programs like the offshore program.",
            actual_output="Yes, there are reports available for Fall 2023, and Enrollment Tracking counts all users, including offshore program users.",
            source_row=15,
            source_evaluate=9,
            source_rand_selection=4,
            source_thread_ts="1671552139.39667",
        ),
        _sampled_conversation(
            conversation_id="vague_but_complete",
            prompt="What updates were made to the enrollment tracking reports?",
            expected_output="The enrollment tracking reports have been updated to include new NRC (new returning continuing) values of First-Year Associate and New Transfer Associate. The specific reports are highlighted in the provided screenshot.",
            actual_output="The enrollment tracking reports were updated to include new NRC values for associate users.",
            source_row=9,
            source_evaluate=3,
            source_rand_selection=11,
            source_thread_ts="1695427890.32645",
        ),
        _sampled_conversation(
            conversation_id="multi_fact_missing_1",
            prompt="Where can I find comparison reports for Q1 2021 vs Q1 2022?",
            expected_output="The reporting site only provides options for Q1 2022 vs Q1 2023. For a 5-year analysis, it is recommended to use the quarterly census data for trend reporting. The tracking reports are not based on quarterly snapshots but on point-in-time comparisons. Custom reports can be built using snapshots going back to 2005, but a specific point in time must be chosen for comparison. It is suggested to check the reports published by the analytics team.",
            actual_output="The reporting site only provides options for Q1 2022 vs Q1 2023. For a 5-year analysis, it is recommended to use the quarterly census data for trend reporting. The tracking reports are not based on quarterly snapshots but on point-in-time comparisons. Custom reports can be built using snapshots going back to 2005, but a specific point in time must be chosen for comparison.",
            source_row=14,
            source_evaluate=2,
            source_rand_selection=5,
            source_thread_ts="1676322629.92399",
        ),
        _sampled_conversation(
            conversation_id="multi_fact_missing_2",
            prompt="Where can I find comparison reports for Q1 2021 vs Q1 2022?",
            expected_output="The reporting site only provides options for Q1 2022 vs Q1 2023. For a 5-year analysis, it is recommended to use the quarterly census data for trend reporting. The tracking reports are not based on quarterly snapshots but on point-in-time comparisons. Custom reports can be built using snapshots going back to 2005, but a specific point in time must be chosen for comparison. It is suggested to check the reports published by the analytics team.",
            actual_output="The reporting site only provides options for Q1 2022 vs Q1 2023. For a 5-year analysis, it is recommended to use the quarterly census data for trend reporting. The tracking reports are not based on quarterly snapshots but on point-in-time comparisons.",
            source_row=14,
            source_evaluate=2,
            source_rand_selection=5,
            source_thread_ts="1676322629.92399",
        ),
        _sampled_conversation(
            conversation_id="multi_fact_missing_3",
            prompt="Where can I find comparison reports for Q1 2021 vs Q1 2022?",
            expected_output="The reporting site only provides options for Q1 2022 vs Q1 2023. For a 5-year analysis, it is recommended to use the quarterly census data for trend reporting. The tracking reports are not based on quarterly snapshots but on point-in-time comparisons. Custom reports can be built using snapshots going back to 2005, but a specific point in time must be chosen for comparison. It is suggested to check the reports published by the analytics team.",
            actual_output="The reporting site only provides options for Q1 2022 vs Q1 2023. For a 5-year analysis, it is recommended to use the quarterly census data for trend reporting.",
            source_row=14,
            source_evaluate=2,
            source_rand_selection=5,
            source_thread_ts="1676322629.92399",
        ),
        _sampled_conversation(
            conversation_id="multi_fact_missing_4",
            prompt="Where can I find comparison reports for Q1 2021 vs Q1 2022?",
            expected_output="The reporting site only provides options for Q1 2022 vs Q1 2023. For a 5-year analysis, it is recommended to use the quarterly census data for trend reporting. The tracking reports are not based on quarterly snapshots but on point-in-time comparisons. Custom reports can be built using snapshots going back to 2005, but a specific point in time must be chosen for comparison. It is suggested to check the reports published by the analytics team.",
            actual_output="The reporting site only provides options for Q1 2022 vs Q1 2023.",
            source_row=14,
            source_evaluate=2,
            source_rand_selection=5,
            source_thread_ts="1676322629.92399",
        ),
        _sampled_conversation(
            conversation_id="multi_fact_missing_5",
            prompt="Where can I find comparison reports for Q1 2021 vs Q1 2022?",
            expected_output="The reporting site only provides options for Q1 2022 vs Q1 2023. For a 5-year analysis, it is recommended to use the quarterly census data for trend reporting. The tracking reports are not based on quarterly snapshots but on point-in-time comparisons. Custom reports can be built using snapshots going back to 2005, but a specific point in time must be chosen for comparison. It is suggested to check the reports published by the analytics team.",
            actual_output="You may need a different report or a custom extract for this comparison.",
            source_row=14,
            source_evaluate=2,
            source_rand_selection=5,
            source_thread_ts="1676322629.92399",
        ),
        _sampled_conversation(
            conversation_id="numbered_list_formatting_different",
            prompt="How can I find a report to determine the number of users in a program and new signups for Q4 2024?",
            expected_output="To find the desired report, check the User lists on the Enrollment and Records page at https://reports.example.com/enrollment/pages/welcome. The report titled \"Users by Plan as of Today\" lists all active users in a program, including their signup type. You can also run this report by program or plan code. For new users, consider the report at https://reports.example.com/enrollment/new-users. Additionally, the User Enrollment by Department report at https://reports.example.com/enrollment/by-department allows for drilling down by department, providing detailed enrollment data.",
            actual_output="""Recommended sources:
1. User lists on Enrollment and Records: https://reports.example.com/enrollment/pages/welcome
2. Use \"Users by Plan as of Today\" for active users in a program.
3. Signup type indicates new user status.
4. You can run by program or plan code.
5. For new users, consider the report at https://reports.example.com/enrollment/new-users.
6. For more detail, drill into User Enrollment by Department: https://reports.example.com/enrollment/by-department for drilling down by department.
""",
            source_row=3,
            source_evaluate=7,
            source_rand_selection=0,
            source_thread_ts="1724089308.33667",
        ),
        _sampled_conversation(
            conversation_id="json_formatting_significantly_different",
            prompt="Does anyone know of a report that shows older enrollment data for a specific program? I need data from Fall 2021.",
            expected_output="It is suggested to use the Enterprise group page, particularly the Master List for helpful information. For census enrollment data, refer to the following link: https://reports.example.com/institutional-analysis/enrollment",
            actual_output='{"recommendation":"Use the Enterprise group page","resource":"Master List","link":"https://reports.example.com/institutional-analysis/enrollment","note":"Census enrollment data is available here"}',
            source_row=4,
            source_evaluate=8,
            source_rand_selection=0,
            source_thread_ts="1709830802.35589",
        ),
        _sampled_conversation(
            conversation_id="extra_line_breaks_marginal_formatting_difference",
            prompt="Is there a report anywhere that shows users registered for services by department for spring 2023?",
            expected_output="Yes, there are reports available that can be run on demand by department, plan, region, and tier. For a list of users, visit the User Lists page at https://reports.example.com/enrollment/pages/users. For counts by plan, refer to the Enrollment Tracking page at https://reports.example.com/enrollment/pages/tracking. Note that Enrollment Tracking only counts users registered in standard classes, excluding programs like the offshore program.",
            actual_output="Yes, there are reports available that can be run on demand by department, plan, region, and tier. For a list of users, visit the User Lists page at https://reports.example.com/enrollment/pages/users. For counts by plan, refer to the Enrollment Tracking page at https://reports.example.com/enrollment/pages/tracking.\n\nNote that Enrollment Tracking only counts users registered in standard classes, excluding programs like the offshore program.",
            source_row=15,
            source_evaluate=9,
            source_rand_selection=4,
            source_thread_ts="1671552139.39667",
        ),
    ],
)
